#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <cuda_runtime.h>
using namespace nvcuda;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 8;

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    // int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    // int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    int warp_row = blockIdx.y * wmmaM;
    int warp_col = blockIdx.x * wmmaN;
    // printf("x: %d, y: %d, \n", warp_row, warp_col);
    if (warp_row >= M && warp_col >= N)
        return;
    wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += wmmaK) {
        wmma::load_matrix_sync(a_frag, A + i + warp_row * K, K);
        wmma::load_matrix_sync(b_frag, B + warp_col + i * N, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_col + warp_row * N, c_frag, N, wmma::mem_row_major);
}

