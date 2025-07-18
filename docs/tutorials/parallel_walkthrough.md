Auto-Tuning Techniques for Performance Optimization
===================================================
<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/botbw">botbw</a>
</div>

# Overview

`tl.Parallel` is a tile operator which exposes thread-level control to user, 
which might be a bit obscure how it works when taking a first look. 
In this tutorial, we are gonna to do a simple code walk-through to uncover its secrets.

commit: `https://github.com/tile-ai/tilelang/commit/78ee1635faf069657e278dd210f5f0849256dbbc`

# Walkthrough

## IR definition

At python side, `tl.Paralle` is just a simple wrapper using `tvm`'s foreign-language-interface (ffi).

```python
# https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/tilelang/language/parallel.py#L30

def Parallel(*extents: tir.PrimExpr, coalesced_width: Optional[int] = None):
    annotations: Dict[str, Any] = {}
    if coalesced_width is not None:
        annotations.update({"coalesced_width": coalesced_width})
    return _ffi_api.Parallel(extents, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member
```

At c++ side, we defined a op that returns a `ForFrame`

```c++
//  https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/src/ir.cc#L51-L76
ForFrame ParallelFor(Array<PrimExpr> extents,
                     Map<String, ObjectRef> annotations) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [annotations](Array<Var> vars, Array<Range> doms,
                                     Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body =
          For(var, dom->min, dom->extent, ForKind::kParallel, std::move(body),
              /*thread_binding=*/NullOpt, /*annotations=*/annotations);
    }
    return body;
  };
  return ForFrame(n);
}

//  https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/src/ir.cc#L285
TVM_REGISTER_GLOBAL("tl.Parallel").set_body_typed(ParallelFor);
```

Which convert a multi-dim `tl.Parallel` into several nested `tvm.parallel`:

This is exactly same as a for loop definition in [tvm](https://github.com/tile-ai/tvm/blob/db50d4e19e8b04677fff3c32dc7fa4c42799f39a/src/script/ir_builder/tir/ir.cc#L354-L377) but with multi-dim extents and loop variables.

This funciton will be called by [`tvm`](https://github.com/tile-ai/tvm/blob/db50d4e19e8b04677fff3c32dc7fa4c42799f39a/python/tvm/script/parser/core/entry.py#L100) in `PrimFunc` and parsed into an Abstract Syntax Tree (AST).

```python
# elementwise add as an example

# What you write:
for (local_y, local_x) in T.Parallel(64, 64):
    C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]

# What you got
for local_y in T.parallel(64):
  for local_x in T.parallel(64):
      C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
```

## code transformation

The story begins at [`lower.py`](https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/tilelang/engine/lower.py#L195), where a series of tilelang passes get involved.

Here we will only focus on several passes that could transfer `tl.Parallel`, there are also several other related passes that are only for collecting necessary metadata but doesn't make modification.

### ParallelLoopTransformer

This pass will analyze the buffer shape and accessed range, if the buffer is smaller than accessed range, a condition check will be added to avoid illegal memory accessment:
```python
for local_y in T.parallel(4):
  for local_x in T.parallel(4):
    C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
```

### ParallelLoopFuser

This pass will fuse several nested for loop into a single one once some conditions are meet (no fragment accessment, 2^n extent...)

```python
# before
for local_y in T.parallel(64):
  for local_x in T.parallel(64):
      C_shared[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]

# after
for local_y_local_x_fused in T.parallel(4096):
  C_shared[local_y_local_x_fused % 4096 // 64, local_y_local_x_fused % 64] = A_shared[local_y_local_x_fused % 4096 // 64, local_y_local_x_fused % 64] + B_shared[local_y_local_x_fused % 4096 // 64, local_y_local_x_fused % 64]
```

### LayoutInference

This is where a tvm `ForNode` got [transferred](https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/src/transform/layout_inference.cc#L325) into a tilelang's `ParallelForOp` and run [`InferLayout`](https://github.com/tile-ai/tilelang/blob/78ee1635faf069657e278dd210f5f0849256dbbc/src/transform/layout_inference.cc).

