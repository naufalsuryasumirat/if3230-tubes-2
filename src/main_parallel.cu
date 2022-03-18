#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "serial.c"

#define TRACE(x) printf("\n-------------------%d-------------------\n", x)

#define BLOCK 32

// Testing copied matrix
__global__ void parallel_printing(Matrix *d_kernel) {
  	printf("%d %d\n", d_kernel->row_eff, d_kernel->col_eff);
	for (int i = 0; i < d_kernel->row_eff; i++) {
		for (int j = 0; j < d_kernel->col_eff; j++) {
			printf("%d ", d_kernel->mat[i][j]);
			if (j == d_kernel->col_eff - 1) printf("\n");
		}
	}
}

// Parallelized Convolution
__global__ void parallel_convolution(Matrix *kernel, Matrix *target, Matrix *out, int *out_row_eff, int *out_col_eff) {
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i = tid / *out_col_eff;
	int j = tid % *out_col_eff;

	if (i < *out_row_eff && j < *out_col_eff) {
		int temp = 0;
		for (int k_i = 0; k_i < kernel->row_eff; k_i++) {
			for (int k_j = 0; k_j < kernel->col_eff; k_j++) {
				temp += kernel->mat[k_i][k_j] * target->mat[i + k_i][j + k_j];
			}
		}
		out->mat[i][j] = temp;
	}
}

// Device function to swap two values
__device__ void swap(int *a, int *b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

// Brick-sort parallelized (Odd-Even Sort)
__global__ void brick(int* arr,int even,int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int id_1 = tid * 2;
	int id_2 = id_1 + 1;
	int id_3 = id_1 + 2;

	if (!even && ((id_2) < n)) {
		if (arr[id_1] > arr[id_2]) {
			swap(&arr[id_1], &arr[id_2]);
		}
	}

	if (even && ((id_3) < n)) {
		if (arr[id_2] > arr[id_3]) {
			swap(&arr[id_2], &arr[id_3]);
		}
	}
}

// Calculate time difference
timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	int *d_orow, *d_ocol;
	Matrix *d_kernel, *d_target, *d_out;
	timespec time_start, time_end, elapsed;
	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);

	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];

	// Using Matrix struct
	cudaMalloc((void **)&d_target, sizeof(Matrix)); // allocating space for target matrix
	cudaMalloc((void **)&d_kernel, sizeof(Matrix)); // allocating space for kernel matrix
	cudaMalloc((void **)&d_out, sizeof(Matrix)); // allocating space for output matrix

	// Copying kernel matrix to device
	cudaMemcpy(d_kernel, &kernel, sizeof(Matrix), cudaMemcpyHostToDevice);

	// Initialize convolution parameters
	int orow = target_row - kernel_row + 1;
	int ocol = target_col - kernel_col + 1;

	Matrix init_out;
	init_matrix(&init_out, orow, ocol); // init row_eff and col_eff of output matrix

	cudaMalloc((void **)&d_orow, sizeof(int));
	cudaMalloc((void **)&d_ocol, sizeof(int));
	cudaMemcpy(d_orow, &orow, sizeof(int), cudaMemcpyHostToDevice); // copying output row size to device
	cudaMemcpy(d_ocol, &ocol, sizeof(int), cudaMemcpyHostToDevice); // copying output col size to device
	cudaMemcpy(d_out, &init_out, sizeof(Matrix), cudaMemcpyHostToDevice); // copying initialized output matrix to device

	int size_process = orow * ocol;
	int thread_per_block = size_process / BLOCK;
	if (size_process % BLOCK > 0) thread_per_block++;
	
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
		// Copy target matrix to device for convolution computation
		cudaMemcpy(d_target, &arr_mat[i], sizeof(Matrix), cudaMemcpyHostToDevice);
		// Parallel convolution using cuda
		parallel_convolution<<<BLOCK,thread_per_block>>>(d_kernel, d_target, d_out, d_orow, d_ocol);
		// Copy output matrix from computation back to host
		cudaError err = cudaMemcpy(&arr_mat[i], d_out, sizeof(Matrix), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
		}
		arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

	int *tosort, sorted[num_targets];
	cudaMalloc((void**)&tosort, num_targets*sizeof(int));
	cudaMemcpy(tosort,arr_range, num_targets*sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < num_targets; i++) {
		brick<<<num_targets/2, 1>>>(tosort, i%2, num_targets);
	}

	cudaMemcpy(sorted,tosort,num_targets*sizeof(int), cudaMemcpyDeviceToHost);

	int median = get_median(sorted, num_targets);
	int floored_mean = get_floored_mean(sorted, num_targets);

	printf("min:%d\nmax:%d\nmedian:%d\nmean:%d\n", 
			sorted[0], 
			sorted[num_targets - 1], 
			median, 
			floored_mean);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_end);
	elapsed = diff(time_start, time_end);

	printf("time spent: %lds:%ldns\n", elapsed.tv_sec, elapsed.tv_nsec);

	// Cleanup
	cudaFree(d_orow);
	cudaFree(d_ocol);
	cudaFree(d_kernel);
	cudaFree(d_target);
	cudaFree(d_out);
	cudaFree(tosort);
	
	return 0;
}