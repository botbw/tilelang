// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm_sp.h
 * \brief Define gemm_sp operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_SP_H_
#define TVM_TL_OP_GEMM_SP_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class GemmSP : public Operator {
 public:
  GemmSP(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

 private:
  std::pair<int, int> ComputeWarpPartition(
      int num_warps, Target target, bool maybe_hopper_wgmma = true) const;

  Array<PrimExpr> call_args;
  tir::Buffer A, B, C, E;
  bool trans_A, trans_B;
  int M, N, K;
  bool clear_accum = false;
  int wg_wait = 0;
  bool completed_ = false;
};

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_GEMM_SP_H_
