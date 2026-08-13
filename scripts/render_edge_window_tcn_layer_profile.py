"""Render a per-stage deployment profile for the 15-window FP32 Edge Window TCN.

The activation values come from ONNX Runtime profiles of the published
combined graph and the published streaming encoder/temporal pair.  FLOPs use
the project's strict convention (two FLOPs per multiply-accumulate) and are
calculated per 1 s hop.  This keeps the figure focused on the scheduling
difference: normal inference encodes 15 windows per hop; the cached schedule
encodes one arriving window and reuses a 15-embedding ring buffer.

Run from ``distrimuse-imu-edge`` with::

    uv run python scripts/render_edge_window_tcn_layer_profile.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import onnx
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = REPO_ROOT.parent / "distrimuse-ds-shared"
WEIGHTS = SHARED_ROOT / "models/imu/weights/edge_window_tcn_wm025"
COMBINED = WEIGHTS / "past7_future7_fp32.onnx"
ENCODER = WEIGHTS / "streaming/past7_future7_encoder_fp32.onnx"
TEMPORAL = WEIGHTS / "streaming/past7_future7_temporal_fp32.onnx"
ASSETS = REPO_ROOT / "experiments/results/edge_window_tcn_context_report_assets"
OUTPUT = ASSETS / "edge-window-tcn-layer-profile"
PROFILE_RECORD = ASSETS / "edge-window-tcn-layer-profile.json"

INK = "#15222d"
MUTED = "#5c6974"
GRID = "#cbd5dc"
SURFACE = "#ffffff"
NORMAL = "#d9e3ed"
CACHE = "#bfe9d7"
SECTION = "#edf3f7"
TOTAL = "#e6f4ef"


@dataclass(frozen=True)
class NodeProfile:
    name: str
    activation_bytes: int
    output_bytes: int
    output_shape: tuple[int, ...]

    @property
    def ping_pong_bytes(self) -> int:
        return self.activation_bytes + self.output_bytes


@dataclass(frozen=True)
class Stage:
    label: str
    normal_shape: str
    cached_shape: str
    normal_bytes: int
    cached_bytes: int
    normal_flops: int
    cached_flops: int
    parameters: int
    section: str | None = None


def _shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def _profile(path: Path) -> list[NodeProfile]:
    """Return activation-only node profiles for one concrete published graph."""
    graph = onnx.load(path).graph
    input_value = graph.input[0]
    input_shape = tuple(
        int(dim.dim_value) if dim.dim_value > 0 else 1
        for dim in input_value.type.tensor_type.shape.dim
    )
    input_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_value.type.tensor_type.elem_type)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    session.run(None, {input_value.name: np.zeros(input_shape, dtype=input_dtype)})
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    finally:
        profile_path.unlink(missing_ok=True)

    nodes: list[NodeProfile] = []
    for event in events:
        args = event.get("args", {})
        if "activation_size" not in args or "output_size" not in args:
            continue
        output_type_shape = args.get("output_type_shape", [])
        if len(output_type_shape) != 1:
            continue
        _, raw_shape = next(iter(output_type_shape[0].items()))
        nodes.append(
            NodeProfile(
                name=str(event["name"]),
                activation_bytes=int(args["activation_size"]),
                output_bytes=int(args["output_size"]),
                output_shape=tuple(int(value) for value in raw_shape),
            )
        )
    if not nodes:
        raise ValueError(f"ONNX Runtime profiling yielded no activation records for {path}")
    return nodes


def _node(nodes: list[NodeProfile], suffix: str) -> NodeProfile:
    matches = [node for node in nodes if node.name.endswith(suffix)]
    if len(matches) != 1:
        names = ", ".join(node.name for node in matches)
        raise ValueError(f"expected one node ending in {suffix!r}, found {len(matches)}: {names}")
    return matches[0]


def _flops_from_macs(macs: int) -> int:
    return 2 * macs


def _stage_flops(*, windows: int) -> dict[str, int]:
    """Strict FLOPs for this width-0.25 model, validated against torchinfo totals.

    The deployed encoder is shared, so its convolution/projection work scales
    with the number of raw windows encoded per hop.  The temporal stack always
    sees the full 15-embedding sequence.
    """
    conv1 = windows * 16 * 312 * (6 * 7 + 1) + windows * 32
    conv2 = windows * 16 * 312 * (16 * 5 + 1) + windows * 32
    conv3 = windows * 32 * 156 * (16 * 5 + 1) + windows * 64
    projection = windows * 24 * (32 + 1)
    tcn_block = 2 * (24 * 15 * (24 * 3 + 1)) + 2 * 48
    head = 48 + 9 * (24 + 1)
    return {
        "conv1": _flops_from_macs(conv1),
        "conv2": _flops_from_macs(conv2),
        "conv3": _flops_from_macs(conv3),
        "projection": _flops_from_macs(projection),
        "tcn_block": _flops_from_macs(tcn_block),
        "head": _flops_from_macs(head),
    }


def _format_flops(value: int) -> str:
    if value == 0:
        return "—"
    return f"{value / 1_000_000:.3f} M"


def _format_kib(value: int) -> str:
    return f"{value / 1024:.2f}"


def _rows() -> tuple[list[Stage], dict[str, int | float]]:
    normal = _profile(COMBINED)
    encoder = _profile(ENCODER)
    temporal = _profile(TEMPORAL)

    normal_nodes = {
        "conv1": _node(normal, "/window_encoder/net/net.0/Conv_kernel_time"),
        "conv2": _node(normal, "/window_encoder/net/net.3/Conv_kernel_time"),
        "pool": _node(normal, "/window_encoder/net/net.6/MaxPool_kernel_time"),
        "conv3": _node(normal, "/window_encoder/net/net.7/Conv_kernel_time"),
        "avg": _node(normal, "/window_encoder/net/net.10/GlobalAveragePool_kernel_time"),
        "projection": _node(normal, "/window_encoder/proj/Gemm_kernel_time"),
        "head": _node(normal, "/head/head.0/LayerNormalization_kernel_time"),
    }
    cached_nodes = {
        "conv1": _node(encoder, "/net/net.0/Conv_kernel_time"),
        "conv2": _node(encoder, "/net/net.3/Conv_kernel_time"),
        "pool": _node(encoder, "/net/net.6/MaxPool_kernel_time"),
        "conv3": _node(encoder, "/net/net.7/Conv_kernel_time"),
        "avg": _node(encoder, "/net/net.10/GlobalAveragePool_kernel_time"),
        "projection": _node(encoder, "/proj/Gemm_kernel_time"),
        "head": _node(temporal, "/head/head.0/LayerNormalization_kernel_time"),
    }
    normal_flops = _stage_flops(windows=15)
    cached_flops = _stage_flops(windows=1)
    ring_buffer_bytes = 15 * 24 * 4

    rows = [
        Stage("Shared CNN encoder", "", "", 0, 0, 0, 0, 0, section="encoder"),
        Stage(
            "Conv1D 6→16, k=7 + BN/ReLU",
            _shape_text(normal_nodes["conv1"].output_shape),
            _shape_text(cached_nodes["conv1"].output_shape),
            normal_nodes["conv1"].ping_pong_bytes,
            cached_nodes["conv1"].ping_pong_bytes,
            normal_flops["conv1"],
            cached_flops["conv1"],
            720,
        ),
        Stage(
            "Conv1D 16→16, k=5 + BN/ReLU",
            _shape_text(normal_nodes["conv2"].output_shape),
            _shape_text(cached_nodes["conv2"].output_shape),
            normal_nodes["conv2"].ping_pong_bytes,
            cached_nodes["conv2"].ping_pong_bytes,
            normal_flops["conv2"],
            cached_flops["conv2"],
            1_328,
        ),
        Stage(
            "MaxPool1D, stride 2",
            _shape_text(normal_nodes["pool"].output_shape),
            _shape_text(cached_nodes["pool"].output_shape),
            normal_nodes["pool"].ping_pong_bytes,
            cached_nodes["pool"].ping_pong_bytes,
            0,
            0,
            0,
        ),
        Stage(
            "Conv1D 16→32, k=5 + BN/ReLU",
            _shape_text(normal_nodes["conv3"].output_shape),
            _shape_text(cached_nodes["conv3"].output_shape),
            normal_nodes["conv3"].ping_pong_bytes,
            cached_nodes["conv3"].ping_pong_bytes,
            normal_flops["conv3"],
            cached_flops["conv3"],
            2_656,
        ),
        Stage(
            "Adaptive average pool",
            _shape_text(normal_nodes["avg"].output_shape),
            _shape_text(cached_nodes["avg"].output_shape),
            normal_nodes["avg"].ping_pong_bytes,
            cached_nodes["avg"].ping_pong_bytes,
            0,
            0,
            0,
        ),
        Stage(
            "Linear projection 32→24",
            _shape_text(normal_nodes["projection"].output_shape),
            _shape_text(cached_nodes["projection"].output_shape),
            normal_nodes["projection"].ping_pong_bytes,
            cached_nodes["projection"].ping_pong_bytes,
            normal_flops["projection"],
            cached_flops["projection"],
            792,
        ),
        Stage(
            "Embedding ring buffer (resident)",
            "—",
            "[1, 24, 15]",
            0,
            ring_buffer_bytes,
            0,
            0,
            0,
        ),
        Stage("Temporal reasoning + head", "", "", 0, 0, 0, 0, 0, section="temporal"),
    ]
    for block_index, dilation in enumerate((1, 2, 4)):
        normal_add = _node(normal, f"/temporal/temporal.{block_index}/Add_kernel_time")
        cached_add = _node(temporal, f"/temporal/temporal.{block_index}/Add_kernel_time")
        rows.append(
            Stage(
                f"Residual TCN block, dilation {dilation} (2× Conv1D + LN)",
                "[1, 24, 15]",
                "[1, 24, 15]",
                normal_add.ping_pong_bytes,
                cached_add.ping_pong_bytes,
                normal_flops["tcn_block"],
                cached_flops["tcn_block"],
                3_600,
            )
        )
    rows.append(
        Stage(
            "Current-token select + LayerNorm + linear 24→9",
            "[1, 9]",
            "[1, 9]",
            normal_nodes["head"].ping_pong_bytes,
            cached_nodes["head"].ping_pong_bytes,
            normal_flops["head"],
            cached_flops["head"],
            273,
        )
    )

    measured_normal_peak = max(stage.normal_bytes for stage in rows)
    measured_cached_peak = max(stage.cached_bytes for stage in rows) + ring_buffer_bytes
    total_normal_flops = sum(stage.normal_flops for stage in rows)
    total_cached_flops = sum(stage.cached_flops for stage in rows)
    total_parameters = sum(stage.parameters for stage in rows)
    if total_parameters != 16_569:
        raise AssertionError(f"unexpected parameter total {total_parameters}")
    return rows, {
        "normal_peak_bytes": measured_normal_peak,
        "cached_peak_bytes": measured_cached_peak,
        "normal_flops": total_normal_flops,
        "cached_flops": total_cached_flops,
        "parameters": total_parameters,
        "ring_buffer_bytes": ring_buffer_bytes,
    }


def _render(rows: list[Stage], totals: dict[str, int | float]) -> None:
    fig, ax = plt.subplots(figsize=(17.2, 10.4))
    ax.axis("off")
    fig.suptitle(
        "Edge Window TCN (15 windows): per-stage activation and compute profile",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.946,
        "FP32 published ONNX graphs · one 1 s hop · normal inference versus embedding-cached streaming",
        fontsize=10.5,
        color=MUTED,
    )
    headers = [
        "Layer / stage",
        "Output shape\nnormal",
        "Output shape\ncached",
        "Peak act. mem.\nnormal (KiB)",
        "Peak act. mem.\ncached (KiB)",
        "FLOPs / hop\nnormal",
        "FLOPs / hop\ncached",
        "Parameters",
    ]
    values: list[list[str]] = []
    cell_colours: list[list[str]] = []
    for stage in rows:
        if stage.section is not None:
            values.append([stage.label] + [""] * (len(headers) - 1))
            cell_colours.append([SECTION] * len(headers))
            continue
        values.append(
            [
                stage.label,
                stage.normal_shape,
                stage.cached_shape,
                _format_kib(stage.normal_bytes),
                _format_kib(stage.cached_bytes),
                _format_flops(stage.normal_flops),
                _format_flops(stage.cached_flops),
                f"{stage.parameters:,}" if stage.parameters else "—",
            ]
        )
        colours = [SURFACE] * len(headers)
        colours[3] = NORMAL
        colours[4] = CACHE
        colours[5] = NORMAL
        colours[6] = CACHE
        cell_colours.append(colours)
    values.append(
        [
            "Total / peak",
            "",
            "",
            f"{_format_kib(int(totals['normal_peak_bytes']))} peak",
            f"{_format_kib(int(totals['cached_peak_bytes']))} peak†",
            _format_flops(int(totals["normal_flops"])),
            _format_flops(int(totals["cached_flops"])),
            f"{int(totals['parameters']):,}*",
        ]
    )
    cell_colours.append([TOTAL] * len(headers))

    table = ax.table(
        cellText=values,
        colLabels=headers,
        colColours=[INK] * len(headers),
        cellColours=cell_colours,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.31, 0.112, 0.112, 0.112, 0.112, 0.095, 0.095, 0.075],
        bbox=[0.02, 0.15, 0.96, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(color="white", weight="bold", fontsize=8.2)
        elif column == 0:
            cell.set_text_props(ha="left")
        if 1 <= row <= len(rows) and rows[row - 1].section is not None:
            cell.set_text_props(weight="bold", ha="left" if column == 0 else "center")
            if column != 0:
                cell.get_text().set_text("")
        if row == len(values):
            cell.set_text_props(weight="bold")

    fig.text(
        0.04,
        0.095,
        "How to read this: normal inference encodes all 15 overlapping raw windows at every hop; cached inference encodes only the arriving window. "
        "The CNN’s second convolution is therefore the normal-schedule bottleneck (585.00 KiB), while the cached schedule peaks at 39.00 KiB in that convolution plus a 1.41 KiB resident embedding buffer (40.41 KiB total).",
        fontsize=9.2,
        color=INK,
        wrap=True,
    )
    fig.text(
        0.04,
        0.045,
        "Activation memory is the real ONNX Runtime activation input plus output for the displayed node; weights, quantization constants, and allocator reservations are excluded. "
        "FLOPs use 2 × MACs. *Parameters are shared by normal and cached execution; †includes the resident [15, 24] float32 embedding ring buffer.",
        fontsize=8.7,
        color=MUTED,
        wrap=True,
    )
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows, totals = _rows()
    _render(rows, totals)
    PROFILE_RECORD.write_text(
        json.dumps(
            {
                "model": "past7_future7_fp32",
                "normal_graph": str(COMBINED.relative_to(SHARED_ROOT)),
                "cached_graphs": [
                    str(ENCODER.relative_to(SHARED_ROOT)),
                    str(TEMPORAL.relative_to(SHARED_ROOT)),
                ],
                "rows": [stage.__dict__ for stage in rows if stage.section is None],
                "totals": totals,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.with_suffix('.svg')} and {PROFILE_RECORD}")


if __name__ == "__main__":
    main()
