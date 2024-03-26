#include "cuda.h"
#include "cuda_fp16.h"
#include "stdio.h"

#define M 4096
#define N 4096
#define K 2048

#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 64

#define WARP_TILE_M 16
#define WARP_TILE_N 16
#define WARP_TILE_K 64

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 8

#define WARP_NUM_M (BLOCK_TILE_M / WARP_TILE_M)
#define WARP_NUM_N (BLOCK_TILE_N / WARP_TILE_N)


#define WARP_PER_BLOCK 16
#define WARP_SIZE 32

#define DATA_BYTE 2
#define A_SHMEM_ELEMENT_NUM (BLOCK_TILE_M * BLOCK_TILE_K) // 
#define B_SHMEM_ELEMENT_NUM (BLOCK_TILE_K * BLOCK_TILE_N)
#define A_SHMEM_SIZE A_SHMEM_ELEMENT_NUM * DATA_BYTE
#define B_SHMEM_SIZE B_SHMEM_ELEMENT_NUM * DATA_BYTE
#define TOTAL_SHMEM_SIZE (A_SHMEM_SIZE + B_SHMEM_SIZE)

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

__global__ void gemm(
    half* A,
    half* B,
    half* C,
    half* D
) {
    extern __shared__ uint8_t smem[];
    half* a_smem = reinterpret_cast<half*>(smem);
    half* b_smem = reinterpret_cast<half*>(smem + A_SHMEM_SIZE);

    half A_frag[4];
    half B_frag[2];
    half C_frag[2][4];
    for (int i = 0; i < 4; i++) {
        A_frag[i] = __float2half(0.0f);
        B_frag[i / 2] = __float2half(0.0f);
        C_frag[0][i] = __float2half(0.0f);
        C_frag[1][i] = __float2half(0.0f);
    }

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t global_A_load_start = blockIdx.x * BLOCK_TILE_M * K + (threadIdx.x / 8) * K + (threadIdx.x % 8) * 8;
    uint32_t global_B_load_start = blockIdx.y * BLOCK_TILE_N + (threadIdx.x / 8) * N + (threadIdx.x % 8) * 8;
    uint32_t shared_A_produce_start = (threadIdx.x / 8) * BLOCK_TILE_K + (threadIdx.x % 8) * 8;
    uint32_t shared_B_produce_start = (threadIdx.x / 8) * BLOCK_TILE_N + (threadIdx.x % 8) * 8;
    uint32_t shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start);
    uint32_t shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start);

    asm volatile (
        "cp.async.ca.shared.global.L2::256B [%0], [%1], 16; \n"
        "cp.async.ca.shared.global.L2::256B [%2], [%3], 16; \n"
        :
        : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start),
          "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
    );
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    uint32_t shared_A_consume_start = (warp_id % WARP_NUM_M) * WARP_NUM_M * BLOCK_TILE_K + lane_id * BLOCK_TILE_K;
    uint32_t shared_B_consume_start = (warp_id / WARP_NUM_M) * BLOCK_TILE_N + lane_id * BLOCK_TILE_K;

    uint32_t shared_A_consume_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_start);
    uint32_t shared_B_consume_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_start);

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%3];\n"
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%2}, [%4];\n"
        : "=r"(*(uint32_t*)(&A_frag[0])), "=r"(*(uint32_t*)(&A_frag[2])), "=r"(*(uint32_t*)(&B_frag[0]))
        : "r"(shared_A_consume_start_addr), "r"(shared_A_consume_start_addr)
    );


    for (int ko = 0; ko < K; ko += BLOCK_TILE_K) {
        for(int m = 0; m < WARP_TILE_M; m += MMA_TILE_M) {
            for(int n = 0; n < WARP_TILE_N; n += MMA_TILE_N) {
                for(int k = 0; k < WARP_TILE_K; k += MMA_TILE_K) {
                    asm volatile (
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {}, {}, {}, {}; \n"
                        : "=r"(*(uint32_t*)(&C_frag[n / MMA_TILE_N][0])),
                          "=r"(*(uint32_t*)(&C_frag[n / MMA_TILE_N][2]))
                        : "r"(*(uint32_t*)(&A_frag[0])),
                          "r"(*(uint32_t*)(&A_frag[2])),
                          "r"(*(uint32_t*)(&B_frag[0])),
                          "r"(*(uint32_t*)(&C_frag[n / MMA_TILE_N][0])),
                          "r"(*(uint32_t*)(&C_frag[n / MMA_TILE_N][2]))
                    );

                    shared_A_consume_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_start + (k + MMA_TILE_K) % WARP_TILE_K);
                    shared_B_consume_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_start + (k + MMA_TILE_K) % WARP_TILE_K * BLOCK_TILE_N + (_n + ((_k == WARP_TILE_K - MMA_TILE_K) ? MMA_TILE_N : 0)) % WARP_TILE_N);
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(*(uint32_t*)(&A_frag[0])),
                          "=r"(*(uint32_t*)(&A_frag[2]))
                        : "r"(shared_A_consume_start_addr)
                    );

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {}, [];\n"
                        : "=r"(*(uint32_t*)(&B_frag[0]))
                        : "r"(shared_B_consume_start_addr)
                    );
                }
            }
        }
        global_A_load_start += BLOCK_TILE_K;
        global_B_load_start += BLOCK_TILE_K * N;
        asm volatile (
            "cp.async.ca.shared.global.L2::256B [%0], [%1], 16; \n"
            "cp.async.ca.shared.global.L2::256B [%2], [%3], 16; \n"
            :
            : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start),
              "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
        );
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();

        shared_A_consume_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_start);
        shared_B_consume_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_start);
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%3];\n"
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%2}, [%4];\n"
            : "=r"(*(uint32_t*)(&A_frag[0])), "=r"(*(uint32_t*)(&A_frag[2])), "=r"(*(uint32_t*)(&B_frag[0]))
            : "r"(shared_A_consume_start_addr), "r"(shared_A_consume_start_addr)
        );
    }

    uint32_t wb_id = blockIdx.x * BLOCK_TILE_M * N + blockIdx.y * BLOCK_TILE_N + (warp_id % 4) * WARP_TILE_M * N + (warp_id / 4) * WARP_TILE_N + (lane_id / 4) * N + (lane_id % 4) * 2;
    *(uint32_t*)(&D[wb_id]) = *(uint32_t*)(&C[0][0]);
    *(uint32_t*)(&D[wb_id + 8 * N]) = *(uint32_t*)(&C[0][2]);
    *(uint32_t*)(&D[wb_id + MMA_TILE_N]) = *(uint32_t*)(&C[1][0]);
    *(uint32_t*)(&D[wb_id + MMA_TILE_N + 8 * N]) = *(uint32_t*)(&C[1][2]);
}

int main() {
    half* h_A, h_B, h_C;
    half* d_A, d_B, d_C;
    half* h_D, d_D;
    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];
    h_D = new half[M * N];

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(half) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_D), sizeof(half) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half(1.5f);
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = __float2half(1.6f);
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C[i * K + j] = __float2half(0.0f);
        }
    }
    
    cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_C), h_C, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    gemm<<<grid, WARP_PER_BLOCK * WARP_SIZE, TOTAL_SHMEM_SIZE>>>(d_A, d_B, d_C, d_D);
    cudaMemcpy(h_D, reinterpret_cast<void*>(d_D), sizeof(half) * M * N, cudaMemcpyDeviceToHost);
}