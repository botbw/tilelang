from dataclasses import dataclass
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang.utils.language import is_fragment
from tvm.ir.base import Node
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import get_buffer_elems


@dataclass
class TransposeWrapper:
    transpose_node: Node

    def infer_layout(self, target: Target, layout_map: dict, thread_nums: int, level: int):
        print(f"{layout_map=} {self.X=}")
        return {}
        # if level != 0:
        #     return {}
        # print(f"{level=} {layout_map=} {T.Fragment(self.XT.shape, forward_fn=shared_16x16_to_mma_32x8_layout_rs_b)=}")
        # return {
        #     self.XT: T.Fragment(self.XT.shape, forward_fn=shared_16x16_to_mma_32x8_layout_rs_b)
        # }

    def lower(self, target: Target, layout_map: dict, thread_nums: int, thread_var: tir.Var):
        assert is_fragment(self.X) and is_fragment(self.XT)
        X_frag = layout_map[self.X]
        XT_frag = layout_map[self.XT]
        local_size_X = get_buffer_elems(self.X) // X_frag.get_thread_size()
        local_size_XT = get_buffer_elems(self.XT) // XT_frag.get_thread_size()
        print(f"{local_size_X=}")
        assert local_size_X == local_size_XT, f"{local_size_X=} {local_size_XT=}"
        M, N = self.X.shape

        @T.prim_func
        def transpose() -> None:
            pass

        return _Simplify(transpose, inline_let=True)

    @property
    def X(self):
        return self.transpose_node.X

    @property
    def XT(self):
        return self.transpose_node.XT
