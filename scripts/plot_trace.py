#!/usr/bin/env python3
"""Plot a recorded tip trace (from scripts/trace_infinity.py) against its plan.

Left: planned and measured tip path in the base X-Z plane, coloured by time. Right: shape error (distance
to the planned curve) and tracking lag (distance to where the tip should be at that moment). An optional
second trace is drawn faintly for before/after comparisons.

Example
-------
    .venv/bin/python scripts/plot_trace.py docs/motion/infinity_trace.npz
    .venv/bin/python scripts/plot_trace.py new.npz --before old.npz --out compare.png
"""
import argparse
from pathlib import Path

import numpy as np

BG, PANEL, GRID, TEXT, MUTED, GHOST = "#0d1117", "#161b22", "#262c36", "#e6edf3", "#7d8590", "#8b949e"


def shape_error_mm(planned: np.ndarray, measured: np.ndarray, subdivisions: int = 20) -> np.ndarray:
    """Distance (mm) from each measured point to the planned polyline; inputs in metres, (N, 3)."""
    s = np.linspace(0, 1, subdivisions)[:, None, None]
    dense = (planned[:-1][None] + s * (planned[1:] - planned[:-1])[None]).reshape(-1, 3)
    return np.array([np.linalg.norm(dense - m, axis=1).min() for m in measured]) * 1000


def lag_error_mm(measured: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Distance (mm) between measured and time-expected tip positions; inputs in metres, (N, 3)."""
    return np.linalg.norm(measured - expected, axis=1) * 1000


def load(path: Path) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    out["shape_mm"] = shape_error_mm(out["planned"], out["measured"])
    out["lag_mm"] = lag_error_mm(out["measured"], out["expected"])
    return out


def plot(trace: dict, out: Path, before: dict | None = None, title: str = "Gripper tip infinity  ∞") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    origin = trace["planned"][0]
    P, M = (trace["planned"] - origin) * 100, (trace["measured"] - origin) * 100
    t = trace["t"]

    fig = plt.figure(figsize=(14, 6.6), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], wspace=0.18, hspace=0.42,
                          left=0.05, right=0.97, top=0.83, bottom=0.1)
    ax, ax_shape, ax_lag = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for a in (ax, ax_shape, ax_lag):
        a.set_facecolor(PANEL)
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for spine in a.spines.values():
            spine.set_visible(False)

    ax.plot(P[:, 0], P[:, 2], color="#ffffff", lw=7, alpha=0.06, solid_capstyle="round")
    ax.plot(P[:, 0], P[:, 2], color="#c9d1d9", lw=1.2, ls=(0, (4, 3)), alpha=0.8, label="planned")
    if before is not None:
        Mb = (before["measured"] - before["planned"][0]) * 100
        ax.plot(Mb[:, 0], Mb[:, 2], color=GHOST, lw=1.0, alpha=0.35, label="before")
    pts = M[:, [0, 2]].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    for width, alpha in ((9, 0.05), (5, 0.12), (2.2, 1.0)):
        lc = LineCollection(segs, cmap="plasma", linewidth=width, alpha=alpha, capstyle="round")
        lc.set_array(t[:-1])
        ax.add_collection(lc)
    ax.plot([], [], color="#f78166", lw=2.2, label="measured")
    ax.scatter([0], [0], s=90, color=BG, edgecolor="#58a6ff", lw=2, zorder=5)
    ax.annotate("start / end", (0, 0), xytext=(0.6, -0.9), color="#58a6ff", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(P[:, 0].min() - 0.8, P[:, 0].max() + 0.8)
    ax.set_ylim(min(P[:, 2].min(), M[:, 2].min()) - 0.8, max(P[:, 2].max(), M[:, 2].max()) + 0.8)
    ax.set_xlabel("x in base_link (cm)")
    ax.set_ylabel("z in base_link (cm)")
    ax.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=9)
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.12, aspect=40)
    cb.set_label("time (s)", color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    cb.solids.set_alpha(1.0)

    panels = ((ax_shape, "shape_mm", "#3fb950", "Shape error: distance to the planned curve"),
              (ax_lag, "lag_mm", "#f0883e", "Tracking lag: distance to where it should be now"))
    for a, key, color, label in panels:
        err = trace[key]
        top = err.max()
        if before is not None:
            a.plot(before["t"], before[key], color=GHOST, lw=1.0, alpha=0.45)
            a.text(t[0] + 0.5, max(top, before[key].max()) * 1.05, f"before: median {np.median(before[key]):.1f} mm",
                   color=GHOST, fontsize=8.5, va="top")
            top = max(top, before[key].max())
        a.fill_between(t, err, color=color, alpha=0.18, lw=0)
        a.plot(t, err, color=color, lw=1.6)
        a.axhline(np.median(err), color=color, lw=0.8, ls=":", alpha=0.8)
        a.text(t[-1], np.median(err), f" median {np.median(err):.1f} mm", color=color, fontsize=8.5,
               va="bottom", ha="right")
        a.set_title(label, loc="left", fontsize=10.5, color=TEXT, pad=8)
        a.set_ylabel("mm")
        a.set_xlim(t[0], t[-1])
        a.set_ylim(0, top * 1.15)
    ax_lag.set_xlabel("time (s)")

    size = ""
    if "width_cm" in trace:
        size = f"{float(trace['width_cm']):g} x {float(trace['height_cm']):g} cm, {int(trace['loops'])} loops, "
    fig.text(0.05, 0.93, title, fontsize=20, fontweight="bold", color=TEXT)
    fig.text(0.05, 0.885,
             f"{size}{t[-1]:.0f} s  ·  TRAC-IK + min-jerk streaming + gravity/friction feed-forward  ·  "
             f"shape error median {np.median(trace['shape_mm']):.1f} mm, max {trace['shape_mm'].max():.1f} mm",
             fontsize=10.5, color=MUTED)
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path, help=".npz written by scripts/trace_infinity.py")
    parser.add_argument("--before", type=Path, help="optional earlier trace drawn faintly for comparison")
    parser.add_argument("--out", type=Path, help="output image (default: trace path with .png)")
    args = parser.parse_args()
    trace = load(args.trace)
    out = args.out or args.trace.with_suffix(".png")
    plot(trace, out, load(args.before) if args.before else None)
    print(f"shape error median {np.median(trace['shape_mm']):.1f} mm, max {trace['shape_mm'].max():.1f} mm; "
          f"lag median {np.median(trace['lag_mm']):.1f} mm -> {out}")
