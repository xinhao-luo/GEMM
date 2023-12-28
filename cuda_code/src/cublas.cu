#include <cublas_v2.h>
#include <cstdio>
#include <string>
#include <fstream>

#define CHECK_CUBLAS(Expr) { \
    int err = (Expr); \
    if (err != 0) { \
        printf("cuBLAS error %d at line %d\n", err, __LINE__); \
    } \
}

void gemm(cublasHandle_t handle,
          int m,
          int n,
          int k,
          const void *alpha,
          const void *beta,
          cudaDataType_t input_type,
          const void *A,
          const void *B,
          cudaDataType_t output_type,
          void *C,
#if __CUDACC_VER_MAJOR__ >= 11
          cublasComputeType_t compute_type,
#else
          cudaDataType_t compute_type,
#endif
          cublasGemmAlgo_t algo) {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        alpha, B, input_type, n, A, input_type, k,
        beta, C, output_type, n, compute_type, algo));
}

int main(int argc, char **argv) {
    int m, n, k = 0;

    if(argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }
    else {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        // gpu_runtime::load_graph(__host_edges, argv[4], false);

        printf("m = %d, n = %d, k = %d\n", m, n, k);
    }

    // int m = 403394;
    // int n = 16;
    // int k = 22;

    // int m = 5120;
    // int n = 4096;
    // int k = 4096;

    float alpha = 1;
    float beta = 0;

    cudaDataType_t input_type = CUDA_R_32F;
    cudaDataType_t output_type = CUDA_R_32F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    int iter = 200;

    void *A, *B, *C;
    cudaMalloc(&A, m * k * sizeof(float));
    cudaMalloc(&B, k * n * sizeof(float));
    cudaMalloc(&C, m * n * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
         output_type, C, compute_type, algo);

    cudaEventRecord(start);
    for (int i = 0; i < iter; ++i) {
        gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
             output_type, C, compute_type, algo);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    long ops = (long)m * n * k * 2;
    double gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("%f Gops\n", gops);
    
    printf("Time: %fms\n", time_ms/iter); 
    
    std::fstream fp;
	fp.open(rst_file, std::ios::out|std::ios::app);
	fp << time_ms/iter << "," << gops << std::endl;
	fp.close();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}


