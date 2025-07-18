# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T
import tvm

from tilelang.utils.target import determine_target
auto_target = tvm.target.Target(determine_target("auto"))

def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):

    @T.prim_func
    def elem_add(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for (local_y, local_x) in T.Parallel(block_M, block_N):
                C_shared[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add


before = elementwise_add(64, 64, 64, 64, "float32", "float32", 128)
before = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
before = tvm.tir.transform.BindTarget(auto_target)(before)
print(before)

after = tilelang.transform.LayoutInference()(before)
print(after)

