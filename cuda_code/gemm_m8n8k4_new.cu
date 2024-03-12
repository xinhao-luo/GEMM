#include "cuda.h"
#include "cuda_fp16.h"
#include "stdio.h"
// nvcc -o new_gemm new_gemm.cu -code=sm_89 -arch=compute_89 -lineinfo
// ncu --set full --import-source=yes -f -o tmp ./new_gemm

#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 128

#define WARP_PER_BLOCK 16

#define WARP_TILE_M 16
#define WARP_TILE_N 16
#define WARP_TILE_K 128

#define WARP_NUM_M (BLOCK_TILE_M / WARP_TILE_M)
#define WARP_NUM_N (BLOCK_TILE_N / WARP_TILE_N)

#define MMA_TILE_M 16
#define MMA_TILE_N 16
#define MMA_TILE_K 4

#define WARP_SIZE 32

#define SHARED_MEM_OFFSET 16

#define DATA_BYTE 2
#define A_SHMEM_ELEMENT_NUM (BLOCK_TILE_M * (BLOCK_TILE_K + SHARED_MEM_OFFSET))
#define B_SHMEM_ELEMENT_NUM (BLOCK_TILE_K * (BLOCK_TILE_N + SHARED_MEM_OFFSET))
#define A_SHMEM_SIZE A_SHMEM_ELEMENT_NUM * DATA_BYTE 
#define B_SHMEM_SIZE B_SHMEM_ELEMENT_NUM * DATA_BYTE 
#define TOTAL_SHMEM_SIZE (A_SHMEM_SIZE + B_SHMEM_SIZE)

#define M 4096
#define N 4096
#define K 2048

constexpr int iter = 100;

__device__ __forceinline__ uint32_t convert_shmem_ptr_to_uint32(const void* shmem_ptr) {
    uint32_t addr;
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(shmem_ptr)
    );
    return addr;
}

__global__ void ada_hgemm_m64n64k128_fp16_sm89(
    half* A,
    half* B,
    half* C,
    half* D,
    uint32_t m,
    uint32_t n,
    uint32_t k
)
{
    // __shared__ __align__(1 * 1024) uint8_t smem[54 * 1024];
    // extern __shared__ __align__(1 * 1024) uint8_t smem[];
    extern __shared__  uint8_t smem[];
    half *a_smem = reinterpret_cast<half*>(smem);
    half *b_smem = reinterpret_cast<half*>(smem + A_SHMEM_SIZE);
    
    half A_frag[4];
    half B_frag[2];
    half C_frag[2][4];
    for (int i = 0; i < 4; i++) {
        A_frag[i] = __float2half(0.0f);
        C_frag[0][i] = __float2half(0.0f);
        C_frag[1][i] = __float2half(0.0f);
        B_frag[i / 2] = __float2half(0.0f);
    }
    
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    
    uint32_t global_A_load_start = blockIdx.x * BLOCK_TILE_M * K + (threadIdx.x / 16) * K + (threadIdx.x % 16) * 8;
    uint32_t global_B_load_start = blockIdx.y * BLOCK_TILE_N     + (threadIdx.x / 8) * N + (threadIdx.x % 8) * 8;
    uint32_t shared_A_produce_start = (threadIdx.x / 16) * (BLOCK_TILE_K + SHARED_MEM_OFFSET) + (threadIdx.x % 16) * 8;
    uint32_t shared_B_produce_start = (threadIdx.x / 8) * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + (threadIdx.x % 8) * 8;
    uint32_t shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start);
    uint32_t shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start);
    
    asm volatile (
        "cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
        "cp.async.ca.shared.global.L2::256B [%2], [%3], 16;\n"
        : 
        : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start),
          "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
    );

    shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start + (BLOCK_TILE_M / 2) * (BLOCK_TILE_K + SHARED_MEM_OFFSET));
    shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start + (BLOCK_TILE_K / 2) * (BLOCK_TILE_N + SHARED_MEM_OFFSET));
    asm volatile (
        "cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
        "cp.async.ca.shared.global.L2::256B [%2], [%3], 16;\n"
        : 
        : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start + (BLOCK_TILE_M / 2) * K),
          "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start + (BLOCK_TILE_K / 2) * N)
    );
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    // printf("bid.x: %d, bid.y: %d, tid: %d, b_smem: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(*(b_smem + 128 * (64 + 16) - 16)));
    // printf("%d, \n", shared_B_produce_start);

    uint32_t load_shared_A_lane_id;
    uint32_t load_shared_B_lane_id;
    if(lane_id < 16) {
        if(lane_id < 8) 
            load_shared_A_lane_id = (lane_id / 4) * 8 + (lane_id % 4);
        else 
            load_shared_A_lane_id = (lane_id / 12) * 8 + (lane_id % 4);
    } else {
        if(lane_id < 24) 
            load_shared_A_lane_id = 4 + (lane_id / 20) * 8 + (lane_id % 4);
        else 
            load_shared_A_lane_id = 4 + (lane_id / 28) * 8 + (lane_id % 4);
    }
    uint32_t shared_A_consume_start   = (warp_id % WARP_NUM_M) * WARP_TILE_M * (BLOCK_TILE_K + SHARED_MEM_OFFSET) + load_shared_A_lane_id * (BLOCK_TILE_K + SHARED_MEM_OFFSET);
    
    uint32_t shared_B_consume_start;
    load_shared_B_lane_id = lane_id % 4;
    if(lane_id < 16) {
        if(lane_id < 8) 
            shared_B_consume_start   = (warp_id / WARP_NUM_M) * WARP_TILE_N + load_shared_B_lane_id * (BLOCK_TILE_N + SHARED_MEM_OFFSET);
        else 
            shared_B_consume_start   = (warp_id / WARP_NUM_M) * WARP_TILE_N + load_shared_B_lane_id * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + 8;
    } else {
        if(lane_id < 24) 
            shared_B_consume_start   = (warp_id / WARP_NUM_M) * WARP_TILE_N + load_shared_B_lane_id * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + 4;
        else 
            shared_B_consume_start   = (warp_id / WARP_NUM_M) * WARP_TILE_N + load_shared_B_lane_id * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + 4 + 8;
    }

    *(uint64_t*)(&A_frag[0]) = *(uint64_t*)(a_smem + shared_A_consume_start);
    // printf("bid.x: %d, bid.y: %d, tid: %d, A_frag: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(A_frag[3]));
    *(uint64_t*)(&B_frag[0]) = *(uint64_t*)(b_smem + shared_B_consume_start);
    // printf("bid.x: %d, bid.y: %d, tid: %d, B_frag: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(B_frag[0]));

    // #pragma unroll
    for (int ko = 0; ko < K; ko += BLOCK_TILE_K) {

        global_A_load_start = (global_A_load_start + BLOCK_TILE_K)     % (M * K);
        global_B_load_start = (global_B_load_start + BLOCK_TILE_K * N) % (K * N);
        shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start);
        shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start);
        // printf("%d, %d, \n", ko, shared_B_produce_start);
        asm volatile(
            "cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            "cp.async.ca.shared.global.L2::256B [%2], [%3], 16;\n"
            :
            : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start)
              "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
        );

        shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start + (BLOCK_TILE_M / 2) * (BLOCK_TILE_K + SHARED_MEM_OFFSET));
        shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start + (BLOCK_TILE_K / 2) * (BLOCK_TILE_N + SHARED_MEM_OFFSET));
        asm volatile (
            "cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            "cp.async.ca.shared.global.L2::256B [%2], [%3], 16;\n"
            : 
            : "r"(shared_A_produce_start_addr), "l"(A + (global_A_load_start + (BLOCK_TILE_M / 2) * K) % (M * K)),
              "r"(shared_B_produce_start_addr), "l"(B + (global_B_load_start + (BLOCK_TILE_K / 2) * N) % (K * N))
        );
        // printf("bid.x: %d, bid.y: %d, tid: %d, a_smem: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(*(a_smem + shared_A_produce_start)));
        for (int _m = 0; _m < WARP_TILE_M; _m += MMA_TILE_M) {
            for (int _n = 0; _n < WARP_TILE_N; _n += MMA_TILE_N) {

                #pragma unroll
                for (int _k = 0; _k < WARP_TILE_K; _k += MMA_TILE_K) {
                    // printf("mma start, \n");
                    asm volatile (
                        "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};\n"
                        : "=r"(*(uint32_t*)(&C_frag[0])), 
                          "=r"(*(uint32_t*)(&C_frag[2])),
                          "=r"(*(uint32_t*)(&C_frag[4])),
                          "=r"(*(uint32_t*)(&C_frag[6]))
                        : "r"(*(uint32_t*)(&A_frag[0])), 
                          "r"(*(uint32_t*)(&A_frag[2])), 
                          "r"(*(uint32_t*)(&B_frag[0])), 
                          "r"(*(uint32_t*)(&B_frag[2])), 
                          "r"(*(uint32_t*)(&C_frag[0])), 
                          "r"(*(uint32_t*)(&C_frag[2])),
                          "r"(*(uint32_t*)(&C_frag[4])),
                          "r"(*(uint32_t*)(&C_frag[6]))
                    );
                    // printf("bid.x: %d, bid.y: %d, tid: %d, c_reg: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(C_frag[0]));
                    // printf("mma over, \n");
                    // Load A from shared to register
                    *(uint64_t*)(&A_frag[0]) = *(uint64_t*)(a_smem + shared_A_consume_start + (_k + MMA_TILE_K) % WARP_TILE_K);
                    // printf("bid.x: %d, bid.y: %d, tid: %d, A_frag: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(A_frag[3]));
                    *(uint64_t*)(&B_frag[0]) = *(uint64_t*)(b_smem + shared_B_consume_start + ((_k + MMA_TILE_K) % WARP_TILE_K) * (BLOCK_TILE_N + SHARED_MEM_OFFSET));
                    // printf("bid.x: %d, bid.y: %d, tid: %d, B_frag: %f, \n", blockIdx.x, blockIdx.y, threadIdx.x, __half2float(B_frag[0]));
                }
            }
        }
    }
    // // printf("wb_idx: %d\n", wb_idx);
    uint32_t wb_idx = blockIdx.x * BLOCK_TILE_M * N + blockIdx.y * BLOCK_TILE_N + (warp_id % 4) * WARP_TILE_M * N + (warp_id / 4) * WARP_TILE_N;
    if(lane_id < 16) {
        if(lane_id < 8) 
            wb_idx += ((lane_id / 4) * 8 + (lane_id % 4)) * N;
        else 
            wb_idx += (((lane_id / 12) * 8 + (lane_id % 4)) * N + 8);
    } else {
        if(lane_id < 24) 
            wb_idx += (4 + (lane_id / 20) * 8 + (lane_id % 4)) * N;
        else 
            wb_idx += ((4 + (lane_id / 28) * 8 + (lane_id % 4)) * N + 8);
    }
    *(uint64_t*)(&D[wb_idx]) = *(uint64_t*)(&C_frag[0]);
    *(uint64_t*)(&D[wb_idx + 4]) = *(uint64_t*)(&C_frag[4]);
}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(ada_hgemm_m64n64k128_fp16_sm89, cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SHMEM_SIZE);
    half *h_A, *d_A;
    half *h_B, *d_B;
    half *h_C, *d_C;
    half *h_D, *d_D;
    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];
    h_D = new half[M * N];

    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(half) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_D), sizeof(half) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] =  __float2half((1.0 * i + 2.0 * j) / 50000.0);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] =  __float2half(1.0 * (i + 1) * (j + 1) / 50000.0);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C[i * M + j] = __float2half(0.0f);
        }
    }
    cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_C), h_C, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
        ada_hgemm_m64n64k128_fp16_sm89<<<grid, WARP_PER_BLOCK * WARP_SIZE, TOTAL_SHMEM_SIZE>>>(d_A, d_B, d_C, d_D, M, N, K);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    printf("%9.6f\n", (ms / (1.0 * iter)));
    printf("%9.6f TFLOPS\n", ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / 1000.0)) / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    cudaMemcpy(h_D, reinterpret_cast<void*>(d_D), sizeof(half) * M * N, cudaMemcpyDeviceToHost);
    return 0;
}
