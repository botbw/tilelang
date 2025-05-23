// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>

namespace cute {
namespace tl_wgmma_sp {
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename ElementA, typename ElementB,
          typename ElementAccumulator>
class GemmTensorOp {
 public:
  static_assert(num_warp_m % 4 == 0, "num_warp_m must be a multiple of 4");

  // modified form
  // https://github.com/NVIDIA/cutlass/blob/df8a550d3917b0e97f416b2ed8c2d786f7f686a3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L85
  using TileShape_MNK =
      Shape<Int<M / (num_warp_m / 4)>, Int<N / num_warp_n>, Int<K>>;
  using AtomLayoutMNK = Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>;

  using ElementAMmaRaw =
      std::conditional_t<std::is_same<ElementA, float>::value, tfloat32_t, ElementA>;
  using ElementBMma =
      std::conditional_t<std::is_same<ElementB, float>::value, tfloat32_t, ElementB>;

  static constexpr GMMA::Major GmmaMajorA =
      trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB =
      trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using TiledMma = decltype(make_tiled_mma(
      GMMA::ss_op_selector_sparse<ElementAMmaRaw, ElementBMma,
                                  ElementAccumulator, TileShape_MNK, GmmaMajorA,
                                  GmmaMajorB>(),
      AtomLayoutMNK{}));

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementEMmaSparsity = Int<ElementEMma::sparsity>;
  using ElementE = typename ElementEMma::raw_type;

  using SparseConfig =
      cutlass::Sm90GemmSparseConfig<ElementAMma, GmmaMajorA, ElementEMma,
                                    decltype(min(size<2>(TileShape_MNK{}),
                                                 _128{}))>;

  using LayoutA = decltype(SparseConfig::deduce_layoutA());
  using LayoutE = decltype(SparseConfig::deduce_layoutE());

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector_sparse<
               GmmaMajorA, ElementA, decltype(get<0>(TileShape_MNK{})),
               decltype(get<2>(TileShape_MNK{})), ElementAMmaSparsity>());
  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorB, ElementB, decltype(get<1>(TileShape_MNK{})),
               decltype(get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE =
      ComposedLayout<Swizzle<0, 4, 3>,
                     smem_sparse_ptr_flag_bits<ElementEMmaSparsity::value,
                                               sizeof_bits_v<ElementE>>,
                     SmemLayoutAtomE_>;

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
      conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));
  using SmemLayoutE = decltype(tile_to_shape(
      SmemLayoutAtomE{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));

  template <int wg_wait = 0>
  static CUTE_DEVICE void body(ElementA *pA, ElementB *pB,
                               ElementAccumulator *pC, ElementE *pE) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(recast_ptr<ElementAMma>(pA)), SmemLayoutA{});  // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(recast_ptr<ElementBMma>(pB)), SmemLayoutB{});  // (BLK_N,BLK_K)
    Tensor sE = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(recast_ptr<ElementEMma>(pE)), SmemLayoutE{}));  // (BLK_M,BLK_K)

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N)
    Tensor tCsE =
        partition_E(thr_mma, sE(_, _));  // (MMA,MMA_M)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);          // (MMA,MMA_N,MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);          // (MMA,MMA_M,MMA_N)
    Tensor tCrE = make_fragment_like<ElementEMma>(tCsE);  // (MMA,MMA_M,MMA_K)

    Tensor acc =
        make_tensor(make_rmem_ptr(pC),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrE(_, _, k_block)),
           tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(acc);
  }

  template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK,
            class ETensor>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_E(
      TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK> const &mma,
      ETensor &&etensor) {
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    CUTE_STATIC_ASSERT_V(rank(etensor) >= Int<2>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}), get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(etensor, t_tile);  // (PermM,PermK)

    // Tile the tensor for the Atom
    auto e_tile =
        make_tile(make_layout(size<0>(typename TiledMma::AtomShape_MNK{})),
                  make_layout(size<2>(typename TiledMma::AtomShape_MNK{})));
    auto e_tensor =
        zipped_divide(t_tensor, e_tile);  // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    using AtomLayoutE_TV = typename TiledMma::Atom::Traits::ELayout;
    auto tv_tensor =
        e_tensor.compose(AtomLayoutE_TV{}, _);  // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile =
        make_tile(_, make_tile(make_layout(size<1>(mma.thr_layout_vmnk_)),
                               make_layout(size<3>(mma.thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(
        tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class... MArgs, class ETensor>
  CUTE_HOST_DEVICE static constexpr auto partition_E(
      ThrMMA<MArgs...> const &thr_mma, ETensor &&etensor) {
    auto thr_tensor = make_tensor(static_cast<ETensor &&>(etensor).data(),
                                  thrfrg_E(thr_mma, etensor.layout()));

    auto thr_vmk = make_coord(
        get<0>(thr_mma.thr_vmnk_),
        make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
    return thr_tensor(thr_vmk,
                      make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  }
};

}  // namespace tl_wgmma_sp
}  // namespace cute

namespace tl {
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type,
          typename MMA = cute::tl_wgmma_sp::GemmTensorOp<
              M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum,
              A_type, B_type, C_type>,
          typename E_type = typename MMA::ElementEMma::raw_type>
TL_DEVICE void gemm_sp_ss(A_type *pA, B_type *pB, C_type *accum, E_type *pE) {
  static_assert(use_wgmma, "only wgmma is supported for now");
  if constexpr (use_wgmma) {
    MMA::body<wg_wait>(pA, pB, accum, pE);
  } else {
    CUTE_GCC_UNREACHABLE;
  }
}
}  // namespace tl