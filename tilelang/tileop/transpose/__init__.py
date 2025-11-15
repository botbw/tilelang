from tilelang import tvm as tvm
from tvm import tir
from tilelang.utils.target import (
    target_is_cuda,)
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang.tileop.transpose.transpose import TransposeWrapper


@tvm_ffi.register_object("tl.Transpose")
class Transpose(Node, Scriptable):
    X: tir.Buffer
    XT: tir.Buffer

    XPtr: tir.PrimExpr
    XTPtr: tir.PrimExpr

    M: int
    N: int

    def infer_layout(self, layout_map: dict, target: Target, thread_nums: int, level: int):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            return TransposeWrapper(self).infer_layout(target, layout_map, thread_nums, level)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            # Now only implement ssr layout
            return TransposeWrapper(self).lower(target, layout_map, thread_nums, thread_var)
        else:
            raise ValueError(f"Unsupported target: {target}")


@tvm_ffi.register_global_func("tl.transpose.infer_layout")
def transpose_infer_layout(transpose: Transpose, target: Target, layout_map: dict,
                           thread_bounds: Range, level: int):
    thread_nums = thread_bounds.extent
    return transpose.infer_layout(target, layout_map, thread_nums, level)


@tvm_ffi.register_global_func("tl.transpose.lower")
def transpose_lower(transpose: Transpose, target: Target, layout_map: dict, thread_bounds: Range,
                    thread_var: tir.Var):
    thread_nums = thread_bounds.extent
    stmt = transpose.lower(target, layout_map, thread_nums, thread_var)
    return stmt
