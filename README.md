## Contents
- `cuda_code`:

    `exp`: 不同版本的CUDA Tensor Core GEMM在不同的size和GNN数据库size下的性能表现实验记录。

    `exp_cublas`: Cublas和Cublas Tensor Core在不同的size和GNN数据库size下在A100和V100上的性能表现实验记录。

    `src`: 

        `gemm_v0.cu`: naive version
        `gemm_v1.cu`: load global memory to shared memory and then load shared memory to fragment
        `gemm_v2.cu`: Double Buffer
        `gemm_v3.cu`: solve memory bank conflict but not complete
    `script_gnn.py`: 脚本程序，测试代码在GNN数据库size下的性能表现。

    `script.py`: 脚本程序，测试代码在不同的常见size下的性能表现。

    `cmd.sh`: 记录了检验代码矩阵乘法的正确性以及测试性能的命令行。

- `tvm_code`: 使用TVM的卷积和矩阵乘法的代码。
- `DGL_Learning`: 

    `test.py`: 使用DGL框架构建和训练GNN模型的测试代码。

## Plan
更多不同策略的内存优化GEMM Code