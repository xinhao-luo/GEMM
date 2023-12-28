#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <algorithm>
#include <cublas_v2.h>

int ITERS = 20;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 8;
// const int WARP_SIZE = 32;

extern __global__ void matmul(float *A, float *B, float *C, int M, int N, int K);

#define MAX(a, b) (a) > (b) ? (a) : (b)

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
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

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    size_t K = atoi(argv[3]);
#ifdef DEBUG
    std::cout << "Debugging using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
#else
    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
#endif
    srand(time(NULL));
    float *hA = (float *)malloc(M * K * sizeof(float));
    float *hB = (float *)malloc(K * N * sizeof(float));
    float *hC = (float *)malloc(M * N * sizeof(float));
    float* h_C1 = (float *)malloc(M * N * sizeof(float));
    
    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        hA[i] = i / 13;
    }

    // 生成B的数据
    for( int i = 0; i < K * N; i++ ) {
        hB[i] = i % 13;
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);
            h_C1[i * N + j] = (float)(0);
        }
    }
    
    float *dA;
    float *dB;
    float *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(float), cudaMemcpyHostToDevice));


    // dim3 dimBlock(32, 2 * MULTI_THREADING, 2);
    // dim3 dimGrid(N / 128, M / 128);

    dim3 dimGrid((N - 1) / wmmaN + 1, (M - 1) / wmmaM + 1);
    dim3 dimBlock(32);
    // dim3 dimGrid((N - 1) / wmmaN + 1, (M - 1) / wmmaM + 1);
    // dim3 dimBlock(32);

    // dim3 dimGrid((N - 1) / 16 + 1, (M - 1) / 16 + 1);
    // dim3 dimBlock(16, 16);
    
#ifndef DEBUG
    int smem_size = (wmmaM * wmmaK + wmmaK * wmmaN) * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warmup
    for (int i = 0; i < ITERS / 20 - 1; ++i)
    {
        matmul<<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
    {
        matmul<<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time: " << double(ms) / ITERS << "ms\n";
    std::cout << (float)M * N * K * 2 / (double(ms) / ITERS) * 1e3 / 1e9 << " Gops\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CUDA kernel is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";
#endif

#ifdef DEBUG
    int smem_size = (wmmaM * wmmaK + wmmaK * wmmaN) * sizeof(float);
    std::cout << "Computing result values...\n";
    matmul<<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Computing results done!\n";
    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < M * N; i++)
    //     std::cout << hC[i] << "\n";
    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH );
    float alpha = 1.0;
    float beta = 0;
    CUDA_CHECK(cudaMemcpy(dC, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
        M, N, K, &alpha, 
        dA, K, dB, N, &beta, dC, M);
    CUDA_CHECK(cudaMemcpy(h_C1, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle);

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(hC[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(hC[i]);
        double rel_err = abs_err / abs_val / dot_length;
        // std::cout << h_C1[col * M + row] << "\n";
        if (rel_err > eps) {
            printf("Error! Matrix[%d][%d]=%.8f, ref=%.8f error term is > %E\n",
                    row, col, hC[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
#endif

    free(hA);
    free(hB);
    free(hC);
    free(h_C1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}