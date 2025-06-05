import torch
import os

from torch.utils.cpp_extension import load
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True

compress_util = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./src/tl_templates/cuda/compress_sm90.cu")
cutlass_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./3rdparty/cutlass")

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


def generate_2_to_4_sparse_tensor(shape, dtype=torch.float32, device='cpu', flat_idx=None):
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    full_tensor = torch.randn(shape, dtype=dtype, device=device)
    # full_tensor = torch.arange(1, prod(shape) + 1, dtype=dtype, device=device).view(shape) * 0.01
    mask = torch.zeros_like(full_tensor, dtype=torch.bool)

    group_count = shape[-1] // 4
    group_shape = shape[:-1] + (group_count, 4)

    reshaped = full_tensor.view(*group_shape)

    for idx in range(reshaped.numel() // 4):
        if flat_idx is None:
            flat_idx = torch.randint(0, 4, (2,), dtype=torch.int64)
            while flat_idx[0] == flat_idx[1]:
                flat_idx[1] = torch.randint(0, 4, (1,), dtype=torch.int64)
        i = idx // group_count
        j = idx % group_count
        mask.view(*group_shape)[i, j, flat_idx[0]] = True
        mask.view(*group_shape)[i, j, flat_idx[1]] = True

    sparse_tensor = full_tensor * mask
    return sparse_tensor


# M, K = 64, 128
M = 64
torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=10000)

for K in [128]:  #range(8, 128, 8):
    print(f"============================{K=}============================")
    A = torch.concat([
        generate_2_to_4_sparse_tensor(
            (M, K // 2), dtype=torch.float16, device='cuda', flat_idx=[0, 1]),
        generate_2_to_4_sparse_tensor(
            (M, K // 2), dtype=torch.float16, device='cuda', flat_idx=[2, 3]),
    ],
     dim=1)

    A_sparse, E = compress_lib.compress_sm90(A)

    print(f"{A.shape=} {A_sparse.shape=} {E.shape=}")
    print(E)
