# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
import torch

import os
import tilelang
from tilelang import tvm as tvm
from torch.utils.cpp_extension import load

tilelang.disable_cache()
# torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=10000)

compress_util = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../src/tl_templates/cuda/compress_sm90.cu")
cutlass_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../3rdparty/cutlass")

compress_lib = load(
    name='compress_lib',
    sources=[
        compress_util,
    ],
    extra_cuda_cflags=[
        '-O2',
        '-std=c++17',
        f'-I{cutlass_repo}/include',
        f'-I{cutlass_repo}/tools/util/include',
        '-arch=sm_90',
    ],
)


def decompose_col_major(index_1d: int, index_nd: list[int]) -> list[int]:
    res = []
    for x in index_nd:
        res.append(index_1d % x)
        index_1d //= x
    # assert index_1d == 0, f"{index_1d=}"
    return res

def global_mapping_row_major(i, j):
    # E global layout: (((_8,_2,_4),2),((_2,_2,_4),2)):(((_16,_2,_256),_1024),((_1,_128,_4),2048))
    # (128, 32)
    i_decomposed = decompose_col_major(i, [8, 2, 4, 2])
    j_decomposed = decompose_col_major(j, [2, 2, 4, 2])
    stride_i = [16, 2, 256, 1024]
    stride_j = [1, 128, 4, 2048]
    i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
    j_offset = sum(j_decomposed[k] * stride_j[k] for k in range(len(j_decomposed)))

    mem_offset = i_offset + j_offset
    return mem_offset

def shared_mapping_row_major(i, j):
    # (128, 16)
    # E shared layout: ((_8,_2,_4, 2),(_2,_2,_4)):((_16,_2,_256, 1024),(_1,_128,_4))
    i_decomposed = decompose_col_major(i, [8, 2, 4, 2])
    j_decomposed = decompose_col_major(j, [2, 2, 4])
    stride_i = [16, 2, 256, 1024]
    stride_j = [1, 128, 4]
    i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
    j_offset = sum(j_decomposed[k] * stride_j[k] for k in range(len(j_decomposed)))

    mem_offset = i_offset + j_offset

    return mem_offset


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
            T.annotate_layout({
                E: T.Layout(E.shape, global_mapping_row_major),
                E_shared: T.Layout(E_shared.shape, shared_mapping_row_major),
            })
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(E[k * block_K // 8, by * block_M], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, B_shared, C_local, E_shared, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def generate_2_to_4_sparse_tensor(shape, dtype=torch.float32, device='cpu', flat_idx_ref=None):
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    full_tensor = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.zeros_like(full_tensor, dtype=torch.bool)

    group_count = shape[-1] // 4
    group_shape = shape[:-1] + (group_count, 4)

    reshaped = full_tensor.view(*group_shape)

    for idx in range(reshaped.numel() // 4):
        if flat_idx_ref is None:
            flat_idx = torch.randint(0, 4, (2,), dtype=torch.int64)
            while flat_idx[0] == flat_idx[1]:
                flat_idx[1] = torch.randint(0, 4, (1,), dtype=torch.int64)
        else:
            flat_idx = flat_idx_ref
        i = idx // group_count
        j = idx % group_count
        mask.view(*group_shape)[i, j, flat_idx[0]] = True
        mask.view(*group_shape)[i, j, flat_idx[1]] = True

    sparse_tensor = full_tensor * mask
    return sparse_tensor

def main():
    torch.random.manual_seed(0)
    M, N, K = 128, 128, 256  
                             # E shared layout: ((_8,_2,_4),(_2,_2,_4)):((_16,_2,_256),(_1,_128,_4))
    block_M, block_N, block_K = 128, 128, 128

    # assert block_K == K, "tiling k is now allowed as meta data is interleaved"
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
    kernel = tilelang.compile(program, out_idx=[-1], pass_configs={
                "tl.disable_tma_lower": True,
            })

    A = generate_2_to_4_sparse_tensor((M, K), dtype=torch.float16, device='cuda', flat_idx_ref=None)
    A_sparse, E = compress_lib.compress_sm90(A)
    print(f'{A=}{A_sparse.shape=} {E.shape=}')
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    C_sp = kernel(A_sparse, E, B).half()
    C = torch.matmul(A, B)
    torch.testing.assert_close(C_sp, C)
    print("pass")


if __name__ == "__main__":
    # print(decompose_col_major(4 + 1 * 8 + 2 * 8 * 2 + 1 * 8 * 2 * 4, [8, 2, 4, 2]))
    main()

# [
# i_j_fused % 2048 // 16 % 64 // 16 * 16 + i_j_fused % 16 % 4 // 2 * 8 + i_j_fused % 2048 // 16 % 8, 
# i_j_fused % 16 // 4 * 4 + i_j_fused % 2048 // 16 % 16 // 8 * 2 + i_j_fused % 16 % 2]