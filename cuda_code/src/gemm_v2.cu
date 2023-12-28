#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <cuda_runtime.h>
using namespace nvcuda;

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 8;

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K)
{
    int warp_row = blockIdx.y * wmmaM;
    int warp_col = blockIdx.x * wmmaN;
    
    // printf("x: %d, y: %d, \n", threadIdx.x, threadIdx.y);
    if (warp_row >= M || warp_col >= N)
        return;

    wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Step 1: Load A to shared memory
    // Double Buffer
    __shared__ float* shared_A[wmmaM * wmmaK * 2];

    int shared_a_offset = threadIdx.x;
    // printf("x: %d, y: %d, \n", shared_a_offset, shared_b_offset);
    // Step 2: Load A and B from global memory to shared memory
    int load_id_a = warp_row * K;
    int shared_flag = 0;
    // printf("x: %d, y: %d, \n", load_id_a, load_id_b);
    if(shared_a_offset < wmmaM * wmmaK && load_id_a < M * K)
        shared_A[shared_a_offset] = A + load_id_a;
    // Ensure all threads in the warp have finished loading A and B before computing
    __syncthreads();

    for (int i = 0; i < K; i += wmmaK) {
        // Step 3: Load shared memory data into fragments And Compute
        wmma::load_matrix_sync(a_frag, shared_A[shared_a_offset], K);
        wmma::load_matrix_sync(b_frag, B + warp_col + i * N, N);

        // Perform matrix multiplication using tensor cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);


        // Load the next tiles of A and B
        shared_a_offset = shared_a_offset * shared_flag + threadIdx.x;

        load_id_a += wmmaK;

        if (shared_a_offset < wmmaM * wmmaK && load_id_a < M * K)
            shared_A[shared_a_offset] = A + load_id_a;
        // Ensure all threads in the warp have finished loading the next tiles
        __syncthreads();
    }

    // Store the fragment results to global memory
    int output_id = (warp_row * N) + warp_col;
    if(output_id < M * N)
        wmma::store_matrix_sync(C + output_id, c_frag, N, wmma::mem_row_major);
}
