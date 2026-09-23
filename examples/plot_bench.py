# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw docs/images/throughput.png from a bench JSON (`examples/bench.py --all --json ...`).

    python examples/plot_bench.py docs/bench.json docs/images/throughput.png

One bar per backend, the best configuration of each (log scale), labelled with the
throughput and the ratio to JSBSim on one CPU core.  Needs matplotlib.
"""
from __future__ import annotations

import json
import sys


def best(rows, pred):
    """Best (rate, label) among rows whose backend matches; label = config (+ backend detail)."""
    got = [(r["rate"], r["config"] + (", " + r["backend"].split(" ", 2)[2]
                                       if r["backend"].startswith("torch CPU") else ""))
           for r in rows if pred(r["backend"])]
    return max(got) if got else None


def main(src: str, dst: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(src, encoding="utf-8"))
    rows, meta = d["rows"], d["meta"]
    base = next(r["rate"] for r in rows
                if r["backend"].endswith("(run only)") and r["config"] == "1 process")
    bars = [
        ("JSBSim, 1 CPU core", (base, "1 process")),
        ("JSBSim, all CPU cores", best(rows, lambda b: b.endswith("(run only)"))),
        ("torch CPU backend", best(rows, lambda b: b.startswith("torch CPU"))),
        ("torch GPU, eager", best(rows, lambda b: b == "torch GPU eager")),
        ("torch GPU + CUDA graph", best(rows, lambda b: b == "torch GPU + CUDA graph")),
        ("CUDA kernel", best(rows, lambda b: b == "CUDA kernel")),
    ]
    bars = [(name, v) for name, v in bars if v]
    colors = ["#e8710a" if n == "CUDA kernel" else "#9aa0a6" if n.startswith("JSBSim")
              else "#8ab4f8" for n, _ in bars]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=150)
    y = list(range(len(bars)))[::-1]
    vals = [v[0] for _, v in bars]
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n({v[1]})" for n, v in bars], fontsize=8.5)
    ax.set_xlim(min(vals) / 3, max(vals) * 60)
    for yi, (n, (v, _)) in zip(y, bars):
        txt = f"{v / 1e9:.2f} G/s" if v >= 1e9 else (f"{v / 1e6:.1f} M/s" if v >= 1e6 else f"{v / 1e3:.0f} k/s")
        r = v / base
        r = float(f"{r:.3g}")                     # 3 significant figures, like the tables
        ax.text(v * 1.15, yi, f"{txt}   {r:,.0f}x", va="center", fontsize=9,
                fontweight="bold" if n == "CUDA kernel" else "normal")
    ax.set_xlabel("aircraft-frames per second (1 frame = 1/120 s), log scale")
    gpu = (meta.get("gpu") or "").replace("NVIDIA GeForce ", "")
    cpu = (meta.get("cpu") or "").replace(" 16-Core Processor", "").replace("AMD ", "")
    ax.set_title(f"F-16 flight model throughput (float32, {meta.get('substeps', 6)} frames per call)"
                 f"\n{gpu}  /  {cpu}", fontsize=9.5)
    ax.grid(axis="x", which="major", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(dst)
    print(f"-> {dst}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("docs/bench.json", "docs/images/throughput.png")))
