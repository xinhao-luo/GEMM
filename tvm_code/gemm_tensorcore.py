import tvm
from tvm import te
import numpy as np
from tvm.contrib import nvcc

# TensorCore shape
block_size = 16

M = 2708
K = 16
N = 7
# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k].astype("float32") * B[k, y].astype("float32"), axis=k), name="C")

target = "cuda"
# Default schedule
s = te.create_schedule(C.op)

# Designate the memory hierarchy
A_shared = s.cache_read(A, "shared", [C])
A_fragment  = s.cache_read(A_shared, "wmma.matrix_a", [C])
B_shared = s.cache_read(B, "shared", [C])
B_fragment  = s.cache_read(B_shared, "wmma.matrix_b", [C])
C_fragment = s.cache_write(C, "wmma.accumulator")
print(tvm.lower(s, [A, B, C], simple_mode=True))

def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=1, strides=[64, 1]
    )
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=1, strides=[N, 1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                n,
                n,
                BA.elem_offset//1024*4 + (BA.elem_offset//16) % 4,
                BC.access_ptr("w"),
                N,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    B = te.placeholder((n, n), name="B", dtype="float32")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum(A[ii, k].astype("float32") * B[k, jj].astype("float32"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=256, strides=[16, 1]
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=16, strides=[64, 1]
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator", data_alignment=32, offset_factor=1, strides=[64, 1]
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment", 
                    BC.data, n, n, n, BC.elem_offset//1024*4+(BC.elem_offset//16)%4, 0.0
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset//1024*4+(BC.elem_offset//16)%4,
                    BA.data,
                    BA.elem_offset//256,
                    BB.data,
                    BB.elem_offset//16,
                    BC.data,
                    BC.elem_offset//1024*4+(BC.elem_offset//16)%4,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

def intrin_wmma_load_matrix_a(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=1, strides=[64, 1])
    C = te.compute((n, n), lambda i, j: A[i, j].astype("float32"), name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=1, strides=[16, 1])


    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset//256,
                BA.access_ptr("r"),
                64,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_b(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=1, strides=[128, 1])
    C = te.compute((n, n), lambda i, j: A[i, j].astype("float32"), name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=1, strides=[64, 1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset//16,
                BA.access_ptr("r"),
                128,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

# Define tiling sizes
block_row_warps = 256
block_col_warps = 128
warp_row_tiles = 64
warp_col_tiles = 64
warp_size = 32
chunk = 4

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")


 # Split each axis into block axis, thread axis, and inner axis
x, y = s[C].op.axis
xo, xi = s[C].split(x, factor=block_row_warps)
yo, yi = s[C].split(y, factor=block_col_warps)
xio, xii = s[C].split(xi, factor=warp_row_tiles)
yio, yii = s[C].split(yi, factor=warp_col_tiles)
xiio, xiii = s[C].split(xii, factor=block_size)
yiio, yiii = s[C].split(yii, factor=block_size)
s[C].reorder(xo, yo, xio, yio, xiio, yiio, xiii, yiii)

xyio = s[C].fuse(xio, yio)
s[C].bind(xo, block_x)
s[C].bind(yo, block_y)
s[C].bind(xyio, thread_y)

# Schedule local computation
s[C_fragment].compute_at(s[C], xyio)
xbf, ybf = s[C_fragment].op.axis
xbfo, xbfi = s[C_fragment].split(xbf, factor=block_size)
ybfo, ybfi = s[C_fragment].split(ybf, factor=block_size)
(zbf,) = s[C_fragment].op.reduce_axis
zbfo, zbfi = s[C_fragment].split(zbf, factor=block_size)
s[C_fragment].reorder(zbfo, xbfo, ybfo, xbfi, ybfi, zbfi)

# Move intermediate computation into each output compute tile
zbfoo, zbfoi = s[C_fragment].split(zbfo, factor=chunk)
s[A_fragment].compute_at(s[C_fragment], zbfoi)
s[B_fragment].compute_at(s[C_fragment], zbfoi)

# Schedule for share memory
s[A_shared].compute_at(s[C_fragment], zbfoo)
s[B_shared].compute_at(s[C_fragment], zbfoo)
xaf, yaf = s[A_fragment].op.axis
xafo, xafi = s[A_fragment].split(xaf, factor=block_size)
xwf, ywf = s[B_fragment].op.axis
ywfo, ywfi = s[B_fragment].split(ywf, factor=block_size)
s[B_fragment].reorder(ywfo, xwf, ywfi)
xas, yas = s[A_shared].op.axis
xwas = s[A_shared].fuse(xas, yas)
xwaso, xwasi = s[A_shared].split(xwas, 256)
xws, yws = s[B_shared].op.axis
xwws = s[B_shared].fuse(xws, yws)
xwwso, xwwsi = s[B_shared].split(xwws, 256)
xwasoo, xwasoi = s[A_shared].split(xwaso, 8)
xwasio, xwasii = s[A_shared].split(xwasi, 8)
xwwsoo, xwwsoi = s[B_shared].split(xwwso, 8)
xwwsio, xwwsii = s[B_shared].split(xwwsi, 8)
s[A_shared].bind(xwasoi, thread_y)
s[A_shared].bind(xwasio, thread_x)
s[A_shared].vectorize(xwasii)
s[B_shared].bind(xwwsoi, thread_y)
s[B_shared].bind(xwwsio, thread_x)
s[B_shared].vectorize(xwwsii)


###############################################################################
# Lowering Computation to Intrinsics
# ----------------------------------
# The last phase is to lower the computation loops down to TensorCore hardware intrinsics
# by mapping the GEMM to tensor intrinsics
s[A_fragment].tensorize(xafi, intrin_wmma_load_matrix_a("wmma.matrix_a"))
s[B_fragment].tensorize(xwf, intrin_wmma_load_matrix_b("wmma.matrix_b"))
s[C].tensorize(xiii, intrin_wmma_store_matrix())
s[C_fragment].tensorize(xbfi, intrin_wmma_gemm())
print(tvm.lower(s, [A, B, C], simple_mode=True))

dev = tvm.cuda(0)
if nvcc.have_tensorcore(dev.compute_version):
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        func = tvm.build(s, [A, B, C], "cuda")
        module = func.imported_modules[0]  
        with open("codegen_tvm_gemm_tensorcore", "w") as f:
            f.write(module.get_source())
    a_np = np.random.uniform(size=(M, K)).astype(A.dtype)
    b_np = np.random.uniform(size=(K, N)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print("conv2d with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

