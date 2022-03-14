#include <stdio.h>
#include <stdlib.h>
#include "convolution.cuh"
#include "serial.h"

// Cuda convolution library implementation
// initialize matrix *out outside of function first
__global__ void parallel_convolution(Matrix *kernel, Matrix *target, Matrix *out) {
    
}