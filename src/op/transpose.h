/*!
 * \file tl/op/tranapose_py.h
 * \brief Transpose operators for tensor computations
 */

#ifndef TVM_TL_OP_TRANSPOSE_H_
#define TVM_TL_OP_TRANSPOSE_H_

#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

class TransposeNode : public TileOperatorNode {
public:
  tir::Buffer X, XT;
  PrimExpr Xptr, XTptr;
  int M, N;
  mutable bool completed_ = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Transpose", TransposeNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TransposeNode>()
        .def_ro("X", &TransposeNode::X)
        .def_ro("XT", &TransposeNode::XT)
        .def_ro("Xptr", &TransposeNode::Xptr)
        .def_ro("XTptr", &TransposeNode::XTptr)
        .def_ro("M", &TransposeNode::M)
        .def_ro("N", &TransposeNode::N)
        .def_ro("completed_", &TransposeNode::completed_);
  }

  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class Transpose : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Transpose, TileOperator,
                                             TransposeNode);
  TVM_DLL Transpose(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_TRANSPOSE_H_
