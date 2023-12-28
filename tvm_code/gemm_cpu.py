import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "llvm"
dev = tvm.device(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 100
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)

print(tvm.lower(s, [A, B, C], simple_mode=True))

bn = 32
kfactor = 4
s = te.create_schedule(C.op)

# We have to re-write the algorithm slightly.
packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
)
C = te.compute(
    (M, N),
    lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
    name="C",
)
s = te.create_schedule(C.op)
 
CC = s.cache_write(C, "global")
 
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
 
s[CC].compute_at(s[C], no)
 
mc, nc = s[CC].op.axis
 
(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)
s[CC].unroll(ki)
 
# parallel
s[C].parallel(mo)
 
bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)
 
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func
 
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
 
evaluator = func.time_evaluator(func.entry_name, dev, number=50)
opt6_time = evaluator(a, b, c).mean
print("Opt6: %f" % opt6_time)
print(tvm.lower(s, [A, B, C], simple_mode=True))