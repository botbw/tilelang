"""The language interface for tl programs."""
from __future__ import annotations

import tilelang.language as T
from tvm import tir


def transpose(
    X: tir.Buffer | tir.Var,
    XT: tir.Buffer | tir.Var,
):

    def legalize_arguments(arg: tir.Buffer | tir.Var):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    X = legalize_arguments(X)
    XT = legalize_arguments(XT)
    M, N = X.shape
    M_, N_ = XT.shape

    assert M == M_ and N == N_, f"Shape doesn't match, {X.shape=} {XT.shape=}"
    Xptr = X.access_ptr("r")
    XTptr = XT.access_ptr("rw")
    return tir.call_intrin("handle", tir.op.Op.get("tl.transpose"), Xptr, XTptr, M, N)
