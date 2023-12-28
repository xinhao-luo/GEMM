# import d2ltvm
import numpy as np
import timeit
import tvm
from tvm import te
import argparse
from typing import Tuple
import re
import os 

def parse_culaunch_config(
    tvm_ir: tvm.IRModule,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    tvm_ir_str = str(tvm_ir)
    rule = 'attr \[IterVar\((\w+\.\w+): int32, \(nullptr\), "ThreadIndex", "(\w+\.\w+)"\)\] "thread_extent" = (\d+)'
    res = re.findall(rule, tvm_ir_str)
    size = {
        "blockIdx.x": 1,
        "blockIdx.y": 1,
        "blockIdx.z": 1,
        "threadIdx.x": 1,
        "threadIdx.y": 1,
        "threadIdx.z": 1,
    }
    for r in res:
        if r[0] == r[1]:
            size[r[0]] = int(r[2])
    return (
        (size["blockIdx.x"], size["blockIdx.y"], size["blockIdx.z"]),
        (size["threadIdx.x"], size["threadIdx.y"], size["threadIdx.z"]),
    )
    
parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=1024, help="vertex size")
parser.add_argument("--N", type=int, default=1024, help="input feature size")
parser.add_argument("--K", type=int, default=1024, help="output feature size")
parser.add_argument("--device", type=str, default="0", help="device id")
# parser.add_argument("--d_type", type=str, default="float32", help="data type")
parser.add_argument("--rst_file", type=str, default="rst", help="result file")
parser.add_argument("--codegen_file", type=str, default="codegen", help="codegen file")
parser.add_argument("--cuda_config_file", type=str, default="", help="cuda config file")

parser.add_argument("--block_size", type=int, default=16, help="block size")
parser.add_argument("--tx", type=int, default=8, help="tile x")
parser.add_argument("--ty", type=int, default=4, help="tile y")
parser.add_argument("--tk", type=int, default=16, help="tile k")

parser.add_argument("--debug", action="store_true", help="print debug info")
parser.add_argument("--cublas", action="store_true", help="test cublas")

args = parser.parse_args()

print(args)

M = 1024
N = 1024
K = 1024

# M = args.M
# N = args.N
# K = args.K

def split(stage, axis, factors):
    """Split an axis by a list of factors in a reverse order
    """
    axes = []
    for f in reversed(factors):
        axis, x = stage.split(axis, f)
        axes.append(x)
    return list(reversed(axes+[axis]))

def bind_thread(stage, axes, tags):
    """Bind a list of axes to thread axes
    """
    for axis, tag in zip(axes, tags):
        stage.bind(axis, te.thread_axis(tag))

# block_size = 16  # the number of threads for one dimension in a thread block.
# tx, ty, tk = 8, 4, 16  # tile sizes for one CUDA thread
block_size = args.block_size
tx = args.tx
ty = args.ty
tk = args.tk

# def matmul(n, m, l):
#     """Return the computing expression of matrix multiplication
#     A : n x l matrix
#     B : l x m matrix
#     C : n x m matrix with C = A B
#     """
#     k = te.reduce_axis((0, l), name='k')
#     A = te.placeholder((n, l), name='A')
#     B = te.placeholder((l, m), name='B')
#     C = te.compute((n, m),
#                     lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
#                     name='C')
#     return A, B, C

A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
k = te.reduce_axis((0, K), 'k')
C = te.compute((M, N),
                lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                name='C')

def matmul_gpu(M, N, K):
    # A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)
    # Create caches
    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[C].op.axis
    xb, xo, xi = split(s[C], x, (block_size, tx))
    yb, yo, yi = split(s[C], y, (block_size, ty))
    s[C].reorder(xb, yb, xo, yo, xi, yi)
    # Note that we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[C], (yb, xb, yo, xo),
                ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # Schedule C_local
    s[C_local].compute_at(s[C], yo)
    yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis
    ko, ki = s[C_local].split(k, tk)
    s[C_local].reorder(ko, ki, yi, xi)
    # Optimize read caches of A and B with cooperative fetching
    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        y, x = s[shared].op.axis
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, yi = s[shared].split(y, nparts=block_size)
        xo, xi = s[shared].split(x, nparts=block_size)
        s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))
    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    return s, (A, B, C)

ctx = tvm.gpu()
ctx.max_shared_memory_per_block/1024
# n = 2048
# M = N = K = n
s, matmul_args = matmul_gpu(M, N, K)
tvm.lower(s, matmul_args, simple_mode=True)

# print(tvm.lower(s, matmul_args, simple_mode=True))

# os._exit()

# target, ctx = 'cuda', tvm.gpu()
target = 'cuda'
mod = tvm.build(s, matmul_args, target)

A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
a_np = np.random.uniform(size=(M, K)).astype(A.dtype)
b_np = np.random.uniform(size=(K, N)).astype(B.dtype)
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(np.zeros((M, N), dtype=B.dtype), ctx)

mod(a, b, c)

if(args.debug):
    np.testing.assert_allclose(
        c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), atol=1e-2)

evaluator = mod.time_evaluator(mod.entry_name, ctx, number=200)
print('gemm: %f ms' % (evaluator(a, b, c).mean * 1e3))


if(args.rst_file != ""):
    with open(args.rst_file, "a+") as f:
        f.write("%f \n" % (evaluator(a, b, c).mean * 1e3))


dev_module = mod.imported_modules[0]        
if(args.codegen_file != ""):
    with open(args.codegen_file, "w") as f:
        f.write(dev_module.get_source())
else:
    if(args.debug):
        print(dev_module.get_source())    
        

# lower mod 
IRMod = tvm.lower(s, matmul_args, simple_mode=True) 
Cuda_Config = parse_culaunch_config(IRMod)

# print(IRMod)

if(args.cuda_config_file != ""):
    with open(args.cuda_config_file, "w") as f:
        # write tuple into cuda_config_file 
        f.write(str(Cuda_Config[0]) + "\n")
        f.write(str(Cuda_Config[1]) + "\n")
else:
    print("Grid: " + str(Cuda_Config[0]))
    print("Block: " + str(Cuda_Config[1]))
        
        
##############################################################################################################
# cublas cuda core test
if(args.cublas):
    cublas_dir = "../src/non_fuse/"
    nvcc_path = "/usr/local/cuda-11.1/bin/nvcc"
    
    nvcc_command = nvcc_path + " " + cublas_dir + "cublas.cu  -lcublas -o " + cublas_dir + "cublas_test"
    print(nvcc_command)
    # run_command = cublas_dir + "cublas_test" + " " + str(M) + " " + str(N) + " " + str(K) + " test \"\" "
    run_command = cublas_dir + "cublas_test" + " " + str(M) + " " + str(N) + " " + str(K)  + "\n"
    print(run_command)
    
    os.system(nvcc_command)
    os.system(run_command)
    