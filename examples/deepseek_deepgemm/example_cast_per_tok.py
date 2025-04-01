import argparse
import torch
import tilelang as tl
import tilelang.language as T

from functools import partial
from typing import Tuple
from tilelang.utils.tensor import map_torch_type, get_dtype_info


# ============ modified from https://github.com/deepseek-ai/DeepGEMM/blob/c57699ac933a93651c34d365797c2d8b41a4765b/tests/test_core.py =============
def ceildiv(a, b):
    return (a + b - 1) // b

def ref_per_token_cast_to_fp8(x: torch.Tensor, tok_dim: int=128, eps: float=1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    b, m, n = x.shape
    x_view = x.view(b, m, -1, tok_dim)
    x_amax = x_view.abs().float().amax(dim=-1).view(b, m, -1).clamp(eps)
    return (x_view * (448.0 / x_amax.unsqueeze(-1))).to(torch.float8_e4m3fn).view(b, m, n), (x_amax / 448.0).view(b, m, -1)
# ========================================================================================================================

def per_token_cast_to_fp8(
    B: int,
    M: int,
    N: int,
    BLOCK_M: int,
    TOK_DIM: int=128,
    in_dtype: str="float32",
    out_dtype: str="e4m3_float8",
    eps: float=1e-4,
):
    assert out_dtype in ["e4m3_float8"]
    torch_in_dype = map_torch_type(in_dtype)
    torch_out_dtype = map_torch_type(out_dtype)
    assert TOK_DIM > 0, f"TOK_DIM should be greater than 0, found {TOK_DIM=}"
    assert N % TOK_DIM == 0, f"N should be divisible by TOK_DIM, found {N=} {TOK_DIM=}"
    out_max = max(get_dtype_info(torch_out_dtype, "max"), abs(get_dtype_info(torch_out_dtype, "min")))

    @T.prim_func
    def main(
        X: T.Tensor((B, M, N), in_dtype),
        Y: T.Tensor((B, M, N), out_dtype),
        Y_scaler: T.Tensor((B, M, N // TOK_DIM), in_dtype)
    ):
        with T.Kernel(
            B,
            T.ceildiv(M, BLOCK_M),
            T.ceildiv(N, TOK_DIM),
            threads=(BLOCK_M, TOK_DIM)
        ) as (b, bm, bn):
            tm = T.get_thread_binding(0)
            tn = T.get_thread_binding(1)
            tok_element = T.alloc_local((1, ), in_dtype)
            tok_abs = T.alloc_local((1, ), in_dtype)
            tok_abs_max = T.alloc_local((1, ), in_dtype)
            tok_element[0] = X[b, bm * BLOCK_M + tm, bn * TOK_DIM + tn]
            tok_abs[0] = T.abs(tok_element[0])
            with T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [T.Cast(in_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        tok_abs[0],
                        True,
                        tok_abs_max[0],
                        tn,
                        dtype="handle",
                    ))
            tok_abs_max[0] = T.max(tok_abs_max[0], eps)
            Y[b, bm * BLOCK_M + tm, bn * TOK_DIM + tn] = (tok_element[0] * (out_max / tok_abs_max[0])).astype(out_dtype)
            if tn == 0:
                Y_scaler[b, bm * BLOCK_M + tm, bn] = tok_abs_max[0] / out_max

    return main


def check_correctness_and_bench(kernel, B, M, N, TOK_DIM=128, bench_ref=True):
    kernel = tl.compile(kernel, out_idx=[1, 2])
    x = torch.randn((B, M, N), device='cuda')
    profiler = kernel.get_profiler()
    output_ref = ref_per_token_cast_to_fp8(x, tok_dim=TOK_DIM)
    output = kernel(x)
    output_ref = output_ref[0].to(torch.float32).reshape(B, M, -1, TOK_DIM) * output_ref[1].unsqueeze(-1)
    output = output[0].to(torch.float32).reshape(B, M, -1, TOK_DIM) * output[1].unsqueeze(-1)
    assert torch.allclose(output, output_ref, atol=1e-1, rtol=1e-1), f"Max diff: {torch.max(torch.abs(output - output_ref))}"
    if bench_ref:
        latency = profiler.do_bench(partial(ref_per_token_cast_to_fp8, tok_dim=TOK_DIM), warmup=500)
        print(f"Torch Latency: {latency} ms")
    latency = profiler.do_bench(kernel, warmup=500)
    print(f"TileLang Latency: {latency} ms\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP8 Cast Example")
    parser.add_argument("--b", type=int, default=16, help="Matrix dimension B")
    parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--tok_dim", type=int, default=128, help="Token dimension")
    args = parser.parse_args()
    B, M, N, TOK_DIM = args.b, args.m, args.n, args.tok_dim

    check_correctness_and_bench(per_token_cast_to_fp8(B, M, N, 1, TOK_DIM=TOK_DIM), B, M, N, TOK_DIM=TOK_DIM)