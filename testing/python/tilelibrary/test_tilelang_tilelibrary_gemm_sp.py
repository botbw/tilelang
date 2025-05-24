# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
import torch

from tilelang import tvm as tvm
from torch.utils.cpp_extension import load

import tilelang.testing

compress_lib = load(
    name='compress_lib',
    sources=[
        'src/tl_templates/cuda/compress_sm90.cu',
    ],
    extra_cuda_cflags=[
        '-O2',
        '-std=c++17',
        '-I/home/wanghaoxuan/cutlass/include',
        '-I/home/wanghaoxuan/cutlass/tools/util/include',
        '-arch=sm_90',
    ],
    build_directory="./build",
)


def matmul_sp(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (M, K)
    A_sparse_shape = (M, K // 2)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K // 2)
    B_shared_shape = (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
            A_sparse: T.Tensor(A_sparse_shape, in_dtype),
            E: T.Tensor((M, K // 8), 'uint8'),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // 8), 'uint8')
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // 8], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, B_shared, C_local, E, False, False)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main():
    M, N, K = 128, 128, 128
    assert K == N
    block_M, block_N, block_K = 128, 128, 128

    program = matmul_sp(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype='float16',
        out_dtype='float32',
        accum_dtype='float32',
        num_stages=1,
        threads=128,
    )

    kernel = tilelang.compile(program, out_idx=[-1])
    A = torch.tensor([0, 1, 2, 0, 0, 0, 3, 4], device='cuda',
                     dtype=torch.float16).repeat([M, K // 8])
    B = torch.eye(K, device='cuda', dtype=torch.float16)
    A_sparse, E = compress_lib.compress_sm90(A)
    print(f"{A_sparse[0]=}\n{E=}")
    C_sp = kernel(A_sparse, E, B)
    C = torch.matmul(A, B).float()
    print(f"{C_sp[0]=}\n{C[0]=}")
    assert torch.allclose(C_sp, C, atol=1e-3), "Sparse GEMM result does not match dense GEMM result"


if __name__ == "__main__":
    main()
