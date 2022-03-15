#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "serial.c"

#define TRACE(x) printf("\n-------------------%d-------------------\n", x)

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

__global__ void parallel_merge(int *n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int left = tid * (NMAX / 16);
	int right = (tid + 1) * (NMAX / 16) - 1;
	
	int mid = left + (right - left) / 2;
	if (low < high) {
		merge_sort(n, left, mid);
		merge_sort(n, mid+1, right);
		merge_array(n, left, mid, right);
	}
}

int main() {
  	int kernel_row, kernel_col, target_row, target_col, num_targets;
	int *d_orow, *d_ocol;
    Matrix *d_kernel, *d_target, *d_out;
	
	clock_t begin = clock();

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
	cudaMemcpy(d_orow, &orow, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ocol, &ocol, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, &init_out, sizeof(Matrix), cudaMemcpyHostToDevice); // copying initialized output matrix to device
	
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
		// Copy target matrix to device for convolution computation
		cudaMemcpy(d_target, &arr_mat[i], sizeof(Matrix), cudaMemcpyHostToDevice);
		parallel_convolution<<<16,16>>>(d_kernel, d_target, d_out, d_orow, d_ocol);
		// Copy output matrix from computation back to host
		cudaError err = cudaMemcpy(&arr_mat[i], d_out, sizeof(Matrix), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
		}
		arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

	// sort the data range array
	merge_sort(arr_range, 0, num_targets - 1);
	
	int median = get_median(arr_range, num_targets);
	int floored_mean = get_floored_mean(arr_range, num_targets);

	// print the min, max, median, and floored mean of data range array
	printf("min:%d\nmax:%d\nmedian:%d\nmean:%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);
			
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time spent: %fs\n", time_spent);
	
	return 0;
}