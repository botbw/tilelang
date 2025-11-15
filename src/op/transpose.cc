/*!
 * \file tl/op/transpose.cc
 * \brief Implementation of Transpose operator
 */

#include "transpose.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

Transpose::Transpose(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<TransposeNode> node = tvm::ffi::make_object<TransposeNode>();
  node->Xptr = args[0];
  node->XTptr = args[1];
  node->X = vmap[GetVarFromAccessPtr(node->Xptr)];
  node->XT = vmap[GetVarFromAccessPtr(node->XTptr)];
  node->M = args[2].as<IntImm>().value()->value;
  node->N = args[3].as<IntImm>().value()->value;
  data_ = std::move(node);
}

TileOperator TransposeNode::Clone() const {
  auto op = tvm::ffi::make_object<TransposeNode>(*this);
  return Transpose(op);
}

Stmt TransposeNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.transpose.lower")) {
    auto prim_func =
        Downcast<PrimFunc>((*f)(tvm::ffi::GetRef<Transpose>(this), T.layout_map,
                                T.target, T.thread_bounds, T.thread_var));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol =
        prim_func->attrs.GetAttr<tvm::ffi::String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<BlockRealizeNode>()) {
      BlockRealize block_realize = Downcast<BlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        BlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // warp with block realize node
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body));
  } else {
    LOG(FATAL) << "No lower function found for transpose";
    return Stmt(); // This line will never be reached due to LOG(FATAL), but
                   // satisfies compiler
  }
}

LayoutMap TransposeNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  LayoutMap results;

  if (const auto f = ffi::Function::GetGlobal("tl.transpose.infer_layout")) {
    results = Downcast<LayoutMap>((*f)(tvm::ffi::GetRef<Transpose>(this),
                                       T.layout_map, T.target, T.thread_bounds,
                                       level));
  } else {
    LOG(FATAL) << "No infer layout function found for transpose";
  }

  return results;
}

TIR_REGISTER_TL_OP(Transpose, transpose)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { TransposeNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
