from tilelang import tvm as tvm
from tilelang.utils.sparse import compress, randn_semi_sparse, randint_semi_sparse
from tilelang.utils.tensor import torch_assert_close, map_torch_type
from tilelang.layout import make_cutlass_metadata_layout
from tilelang.intrinsics.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter

import tilelang.testing
import torch

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    metadata_dtype,
    E_factor,
    num_stages,
    threads,
):
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
            A_sparse: T.Tensor(A_sparse_shape, in_dtype),
            E: T.Tensor((M, K // E_factor), metadata_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // E_factor), metadata_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout({
                E:
                    make_cutlass_metadata_layout(E, mma_dtype=in_dtype, arch="8.0"),
                E_shared:
                    make_cutlass_metadata_layout(
                        E_shared, mma_dtype=in_dtype, arch="8.0"),
            })
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // E_factor], E_shared)
                if trans_A:
                    T.copy(A_sparse[k * block_K // 2, by * block_M], A_shared)
                else:
                    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp_v2(A_shared, E_shared, B_shared, C_frag, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def run_gemm_ss(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    metadata_dtype = 'int32' if ('8' in in_dtype) else 'int16'
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        metadata_dtype,
        SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[in_dtype][metadata_dtype],  # E_factor
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    A, B = generate_dense_input(M, N, K, trans_A, trans_B, in_dtype)

    A_sparse, E = compress(A, transposed=trans_A, block_k=block_K)
    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    torch_assert_close(
        C_sp.to(map_torch_type(out_dtype)).to(torch.float32),
        C.to(map_torch_type(out_dtype)).to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
        base_name="tilelang_sp",
        ref_name="ref_dense",
    )
    print("pass")


def generate_dense_input(M, N, K, trans_A, trans_B, in_dtype):
    is_8bit = "8" in in_dtype
    is_unsigned = "uint" in in_dtype
    is_int = "int" in in_dtype
    if is_int:
        if is_8bit:
            low, high = (0, 4) if is_unsigned else (-2, 2)
        else:
            low, high = (0, 128) if is_unsigned else (-64, 64)
        A = randint_semi_sparse(M, K, low=low, high=high, dtype=map_torch_type(in_dtype), device='cuda', transposed=trans_A)
        B = torch.randint(size=(N, K) if trans_B else (K, N), low=low, high=high, dtype=map_torch_type(in_dtype), device='cuda')
    else:
        A = randn_semi_sparse(M, K, dtype=map_torch_type(in_dtype), device='cuda', transposed=trans_A)
        B = torch.randn((N, K) if trans_B else (K, N), device='cuda', dtype=torch.float32).to(map_torch_type(in_dtype))
    return A, B


def test_gemm_ss():
    # More test case can be found in kernel/test_tilelang_kernel_gemm.py
    # GEMM tests for float16
    run_gemm_ss(512, 1024, 768, False, True, "float16", "float16", "float", 128, 128, 32, 2)
    run_gemm_ss(512, 1024, 768, False, False, "float16", "float16", "float", 128, 128, 32, 2)
    ## run_gemm_ss(512, 1024, 768, True, False, "float16", "float16", "float", 128, 128, 32, 2)
    ## run_gemm_ss(512, 1024, 768, True, True, "float16", "float16", "float", 128, 128, 32, 2)
    # n8 test
    run_gemm_ss(128, 8, 64, False, True, "float16", "float16", "float", 128, 8, 32, 0, 128)

    # int8 test
    run_gemm_ss(128, 128, 128, False, True, "int8", "int32", "int32", 128, 128, 64, 2)
    run_gemm_ss(128, 128, 128, False, False, "int8", "int8", "int32", 128, 128, 64, 2)
    ## run_gemm_ss(128, 128, 128, True, False, "int8", "int8", "int32", 128, 128, 64, 2)
    ## run_gemm_ss(128, 128, 128, True, True, "int8", "int8", "int32", 128, 128, 64, 2)

    # float8 tests
    run_gemm_ss(128, 128, 128, False, True, "float8_e5m2", "float8_e5m2", "float32", 128, 128, 64, 2)

    # tfloat32 test
    # run_gemm_ss(128, 128, 128, False, False, "float", "float", "float32", 128, 128, 32, 2)
    # run_gemm_ss(128, 128, 128, False, True, "float", "float", "float32", 128, 128, 32, 2)
    ##run_gemm_ss(128, 128, 128, True, False, "float", "float", "float32", 128, 128, 32, 2)
    ##run_gemm_ss(128, 128, 128, True, True, "float", "float", "float32", 128, 128, 32, 2)


def matmul_rs(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    A_frag_shape = A_shared_shape

    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_frag)
                T.gemm_v2(A_frag, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-3, rtol=1e-3)


def test_gemm_rs():
    # GEMM tests for float16
    run_gemm_rs(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rs(512, 1024, 768, False, True, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rs(512, 1024, 768, True, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rs(512, 1024, 768, True, True, "float16", "float16", "float16", 128, 256, 32, 2)

    # n8 tests
    run_gemm_rs(128, 8, 32, False, True, "float16", "float16", "float16", 128, 8, 32, 0, 128)

    # int8 tests
    run_gemm_rs(128, 128, 128, False, True, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, False, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, True, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, True, True, "int8", "int8", "int32", 128, 128, 32, 2)

    # float8 tests
    run_gemm_rs(128, 128, 128, True, True, "float8_e5m2", "float8_e5m2", "float32", 128, 128, 32, 2)

    # float32 tests
    run_gemm_rs(128, 128, 128, False, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, False, True, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, True, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rs(128, 128, 128, True, True, "float", "float", "float32", 128, 128, 32, 2)


def matmul_sr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    B_frag_shape = B_shared_shape

    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_shared, B_frag, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_sr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-3, rtol=1e-3)


def test_gemm_sr():
    # GEMM tests for float16
    run_gemm_sr(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_sr(512, 1024, 768, False, True, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_sr(512, 1024, 768, True, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_sr(512, 1024, 768, True, True, "float16", "float16", "float16", 128, 256, 32, 2)

    # n8 tests
    run_gemm_sr(128, 8, 32, False, True, "float16", "float16", "float16", 128, 8, 32, 0, 128)

    # int8 tests
    run_gemm_sr(128, 128, 32, False, True, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 32, False, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 32, True, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 32, True, True, "int8", "int8", "int32", 128, 128, 32, 2)

    # float8 tests
    run_gemm_sr(128, 128, 128, True, True, "float8_e5m2", "float8_e5m2", "float32", 128, 128, 32, 2)

    # float32 tests
    run_gemm_sr(128, 128, 128, False, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 128, False, True, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 128, True, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_sr(128, 128, 128, True, True, "float", "float", "float32", 128, 128, 32, 2)


def matmul_rr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    A_frag_shape = A_shared_shape
    B_frag_shape = B_shared_shape

    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_frag, B_frag, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_rr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-3, rtol=1e-3)


def test_gemm_rr():
    # GEMM tests for float16
    run_gemm_rr(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rr(512, 1024, 768, False, True, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rr(512, 1024, 768, True, False, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rr(512, 1024, 768, True, True, "float16", "float16", "float16", 128, 256, 32, 2)
    run_gemm_rr(512, 1024, 768, False, True, "bfloat16", "bfloat16", "float", 128, 256, 32, 2)
    # n8 tests
    run_gemm_rr(128, 8, 128, False, True, "float16", "float16", "float16", 128, 8, 32, 2)
    run_gemm_rr(128, 8, 128, False, True, "int8", "int8", "int32", 128, 8, 32, 2)

    # int8 tests
    run_gemm_rr(128, 128, 128, False, True, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, False, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, True, False, "int8", "int8", "int32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, True, True, "int8", "int8", "int32", 128, 128, 32, 2)

    # float8 tests
    run_gemm_rr(128, 128, 128, True, True, "float8_e5m2", "float8_e5m2", "float32", 128, 128, 32, 2)

    # float32 tests
    run_gemm_rr(128, 128, 128, False, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, False, True, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, True, False, "float", "float", "float32", 128, 128, 32, 2)
    run_gemm_rr(128, 128, 128, True, True, "float", "float", "float32", 128, 128, 32, 2)


if __name__ == "__main__":
    test_gemm_ss()
