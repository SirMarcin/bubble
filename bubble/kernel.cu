#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include <vector>
#include <limits>
#include <algorithm>

#include <Windows.h>

cudaError_t sortWithCuda(int *a, size_t size, float* time);

typedef long long int64;
typedef unsigned long long uint64;
__host__ int64 GetTimeMs64()
{
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64 ret = li.QuadPart;
	ret -= 116444736000000000LL;

	ret /= 10000; 


	return ret;
}


// Sortowanie w GPU
__global__ void swapOnKernel(int *a, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x * 2;
	int cacheFirst;
	int cacheSecond;
	int cacheThird;

	for (int j = 0; j < size / 2 + 1; j++) {

		if (i + 1 < size) {
			cacheFirst = a[i];
			cacheSecond = a[i + 1];

			if (cacheFirst > cacheSecond) {
				int temp = cacheFirst;
				a[i] = cacheSecond;
				cacheSecond = a[i + 1] = temp;
			}
		}

		if (i + 2 < size) {
			cacheThird = a[i + 2];
			if (cacheSecond > cacheThird) {
				int temp = cacheSecond;
				a[i + 1] = cacheThird;
				a[i + 2] = temp;
			}
		}
		//Synchronizacja w¹tków
		__syncthreads();
	}

}

//Klasyczny bubble sort na CPU
__host__ void bubbleSort(int arr[], int n) {
	// Ÿród³o http://www.algolist.net/Algorithms/Sorting/Bubble_sort
	bool swapped = true;
	int j = 0;
	int tmp;
	while (swapped) {
		swapped = false;
		j++;
		for (int i = 0; i < n - j; i++) {
			if (arr[i] > arr[i + 1]) {
				tmp = arr[i];
				arr[i] = arr[i + 1];
				arr[i + 1] = tmp;
				swapped = true;
			}
		}
	}
}

int main()
{
	srand((unsigned)time(0));
	const int arraySize = 2000;

	//utworzenie wektora (jednowymiarowej tablicy) i zape³nienie go losowymi wartoœciami
	std::vector<int> a(arraySize);
	for (int i = 0; i < arraySize; ++i) {
		a[i] = (rand() * (rand() - 1));
	}
	std::vector<int> b(a);

	float time = 0.0;

	// wywo³anie sortowania na GPU poœrednio poprzez funkcjê hosta
	cudaError_t cudaStatus = sortWithCuda(&a[0], a.size(), &time);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sortWithCuda failed!");
		return 1;
	}

	bool sortingSuccessful = true;
	for (int i = 0; i < a.size() - 1; ++i) {
		if (a[i] > a[i + 1]) {
			sortingSuccessful = false;
			break;
		}
		 printf("%d, ", a[i]);
	}
	printf("\n");

	printf("Time for the GPU: %f ms\n", time);

	if (!sortingSuccessful) {
		printf("Sorting failed.\n");
	}
	//uruchomienie sortowania na CPU i zmierzenie czasu
	int64 stlSortStart = GetTimeMs64();
	bubbleSort(&b[0], b.size());
	int64 stlSortFinish = GetTimeMs64();
	printf("Time for the CPU: %d ms\n",
		(stlSortFinish - stlSortStart));


//zwolnienie zasobów
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//getchar();

	return 0;
}

//uruchomienie kernela z ca³¹ otoczk¹
__host__ cudaError_t sortWithCuda(int *a, size_t size, float* time)
{
	int *dev_a = 0;
	cudaError_t cudaStatus;

	// sprawdzenie czy mamy kompatybiln¹ kartê graficzn¹
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// alokacja pamiêci
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// kopiowanie z hosta na kartê
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// zrobienie tajmera
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//wystartowanie tajmera
	cudaEventRecord(start, 0);
	// uruchomineie sortowania na GPU
	swapOnKernel << <1, size / 2 >> > (dev_a, size);
	// Zatrzymanie tajmera i obliczenie czasu trwania obliczeñ
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time, start, stop);

	// kopiowanie wyniku do osta
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);

	return cudaStatus;
}