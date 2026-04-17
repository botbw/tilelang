from collections.abc import Sequence
from enum import Enum
from pathlib import Path
import subprocess

from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass


class EventType(Enum):
    # initial accepted
    kAccept = "accept"
    # accepted before, and the new inference is aligned with the previous one
    kSame = "same"
    kContain = "contain"
    kMerge = "merge"
    # conflicting inference, cannot accept the new layout
    kReject = "reject"


def _dot_edge_style(level: str, event_type: EventType) -> tuple[str, str, str]:
    if level == "strict":
        style = "solid"
        penwidth = "3.0"
    elif level == "common":
        style = "solid"
        penwidth = "1.5"
    else:
        assert level == "free"
        style = "dashed"
        penwidth = "1.5"

    if event_type == EventType.kAccept:
        color = "#5cb85c"
    elif event_type == EventType.kSame:
        color = "#5bc0de"
    elif event_type == EventType.kContain:
        color = "#f0ad4e"
    elif event_type == EventType.kMerge:
        color = "#428bca"
    else:
        assert event_type == EventType.kReject
        color = "#d9534f"

    return style, penwidth, color


def _write_layout_dag_dot(layout_dag: object, out_dot: Path) -> None:
    op_labels = layout_dag["op_labels"]
    used_op = [False for _ in op_labels]
    events = layout_dag["events"]

    print(f"events: {events}")
    print(f"op_labels: {op_labels}")

    lines: list[str] = []
    lines.append("digraph LayoutInferDAG {")
    lines.append("  rankdir=TB;")
    lines.append("  ordering=out;")
    lines.append("  ranksep=0.7;")
    lines.append("  nodesep=0.35;")
    lines.append('  graph [fontname="Helvetica"];')
    lines.append('  node [fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica"];')

    event_lines = []
    for event in events:
        buffre_name = event["buffer"]
        src = event["src"]
        level = event["level"]
        event_type = EventType(event["event_type"])
        message = event["message"]

        used_op[int(src)] = True

        style, penwidth, color = _dot_edge_style(level, event_type)

        event_lines.append(f'  op_{src} -> {buffre_name} [style="{style}", penwidth="{penwidth}", color="{color}", label="{message}"];')

    prev_op = None
    op_lines = []
    op_connection_lines = []
    for idx, label in enumerate(op_labels):
        if not used_op[idx]:
            continue

        op_name = label
        op_lines.append(f'  op_{idx} [shape=box, label="{op_name}"];')

        if prev_op is not None:
            op_connection_lines.append(f'  op_{prev_op} -> op_{idx} [style="invis", weight="1000", constraint=true];')
        prev_op = idx

    lines.extend(op_lines)
    lines.extend(op_connection_lines)
    lines.extend(event_lines)
    lines.append("}")
    out_dot.write_text("\n".join(lines), encoding="utf-8")


def _try_render_svg(dot_path: Path, svg_path: Path) -> None:
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)


@tir.functor.visitor
class _LayoutDagVisualVisitor(PyStmtExprVisitor):
    def __init__(self, out_dir: str, formats: str | Sequence[str] = "dot"):
        super().__init__()
        if formats is None:
            parsed: list[str] = ["dot"]
        elif isinstance(formats, str):
            formats_str = formats.strip().lower()
            if formats_str == "":
                parsed = ["dot"]
            elif formats_str == "all":
                parsed = ["dot", "svg"]
            elif "," in formats_str:
                parsed = [f.strip() for f in formats_str.split(",") if f.strip()]
            else:
                parsed = [formats_str]
        else:
            parsed = [str(f).strip().lower() for f in formats if str(f).strip()]

        self.formats = set(parsed)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_dag" not in op.annotations:
            return

        dag = op.annotations["layout_dag"]
        dot_path = self.out_dir / f"layout_dag_{self.counter}.dot"
        _write_layout_dag_dot(dag, dot_path)

        if "svg" in self.formats:
            svg_path = self.out_dir / f"layout_dag_{self.counter}.svg"
            try:
                _try_render_svg(dot_path, svg_path)
            except Exception:
                pass

        self.counter += 1


def LayoutDagVisual(out_dir: str = "./tmp/layout_dag", formats: str | Sequence[str] = "dot"):
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutDagVisualVisitor(out_dir=out_dir, formats=formats).visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
