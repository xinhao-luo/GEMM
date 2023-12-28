# debug for correctness

# 常见size和GNN数据库（Cora）size
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v0.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v0.cu src/main.cu -o test && ./test 2708 16 1433
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v1.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v1.cu src/main.cu -o test && ./test 2708 16 1433
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v2.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  -DDEBUG -lcublas -Xcompiler -fopenmp src/gemm_v2.cu src/main.cu -o test && ./test 2708 16 1433

# test performance
nvcc -arch=sm_80  src/gemm_v0.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  src/gemm_v0.cu src/main.cu -o test && ./test 2708 16 1433
nvcc -arch=sm_80  src/gemm_v1.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  src/gemm_v1.cu src/main.cu -o test && ./test 2708 16 1433
nvcc -arch=sm_80  src/gemm_v2.cu src/main.cu -o test && ./test 1024 1024 1024
nvcc -arch=sm_80  src/gemm_v2.cu src/main.cu -o test && ./test 2708 16 1433