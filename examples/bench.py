# SPDX-License-Identifier: GPL-3.0-or-later
"""Throughput: the CUDA kernel, the torch backends, and JSBSim 1.3.0 itself, in one unit.

    python examples/bench.py                 # GPU: CUDA kernel, torch eager, torch + CUDA graph
    python examples/bench.py --cpu           # + torch CPU (batch 1 / 64 / 1,024 / 16,384; 1 thread and default)
    python examples/bench.py --jsbsim        # + JSBSim (1 process ... one per logical core)
    python examples/bench.py --all --json docs/bench.json    # everything + summary table

Unit: **aircraft-frames per second** (one frame = 1/120 s of flight).  One
`step(stick, substeps=6)` call is 6 frames for every aircraft.  JSBSim flies one aircraft
per `FGFDMExec`; one `run()` is one frame.  JSBSim is measured two ways:

    run only      `fdm.run()` in a loop
    with I/O      every frame: write 4 controls (`set_controls`) and read 12 state
                  properties (lat/lon/alt, 3 attitude, 3 body velocity, 3 body rate) --
                  what it costs to actually drive JSBSim from Python

JSBSim is re-trimmed every 100 s of flight (so the fuel never runs out); trim time is not counted.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import platform
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jsbsim_f16_cuda import F16Stick, attach_stick  # noqa: E402

FT, KT = 0.3048, 0.514444
SUB = 6
READS = ("position/lat-geod-rad", "position/long-gc-rad", "position/h-sl-ft",
         "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
         "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",
         "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec")
RETRIM_FRAMES = 12000          # 100 s


# ---------------------------------------------------------------- this port

#: Every aircraft is re-trimmed with a new random stick after this many calls (10 s of
#: flight), outside the timed region, so the fleet keeps manoeuvring inside the envelope
#: instead of flying one fixed input for minutes.
REFRESH_CALLS = 200


def make(n: int, dev: str, dtype) -> tuple:
    """A plant, a stick buffer, and `refresh()` that re-trims everyone with new random sticks
    (random heading, 10,000 - 30,000 ft, 350 - 550 kt; each aircraft holds its own stick)."""
    dyn = F16Stick(n, device=dev, dtype=dtype)
    g = torch.Generator(device=dev).manual_seed(0)
    rnd = lambda *s: torch.rand(*s, device=dev, generator=g, dtype=dtype)
    z = torch.zeros(n, device=dev, dtype=dtype)
    stick = torch.zeros(n, 4, device=dev, dtype=dtype)

    def refresh():
        alt = (10000.0 + 20000.0 * rnd(n)) * FT
        dyn.reset(torch.stack((z, z, -alt), -1), 2 * math.pi * rnd(n), (350.0 + 200.0 * rnd(n)) * KT)
        stick.copy_(torch.stack((0.6 * rnd(n) - 0.3, 0.7 * rnd(n) - 0.2,
                                 0.2 * rnd(n) - 0.1, 0.3 + 0.7 * rnd(n)), -1))

    refresh()
    return dyn, stick, refresh


def rate(fn, n: int, sync, seconds: float, refresh, warm: int = 3, every: int = 10) -> float:
    """One `fn` call = n aircraft x SUB frames.  Time > `seconds` of calls (refreshes excluded),
    return aircraft-frames/s."""
    for _ in range(warm):
        fn()
    sync()
    calls, busy = 0, 0.0
    t0 = time.perf_counter()
    while True:
        fn()
        calls += 1
        if calls % REFRESH_CALLS == 0:
            sync()
            busy += time.perf_counter() - t0
            refresh()
            sync()
            t0 = time.perf_counter()
        if calls % every == 0:
            sync()
            if busy + time.perf_counter() - t0 > seconds:
                return n * SUB * calls / (busy + time.perf_counter() - t0)


def bench_gpu(sizes, dtype, seconds) -> list:
    rows = []
    for n in sizes:
        dyn, stick, refresh = make(n, "cuda", dtype)
        eager = rate(lambda: dyn.step(stick, SUB), n, torch.cuda.synchronize, seconds, refresh)
        # CUDA graph: warm up on a side stream, then capture one call.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                dyn.step(stick, SUB)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dyn.step(stick, SUB)
        graphed = rate(graph.replay, n, torch.cuda.synchronize, seconds, refresh)
        ok = bool(torch.isfinite(dyn.rb.pos_ned).all())
        del dyn, graph
        torch.cuda.empty_cache()
        # CUDA kernel: same plant, `step` routed through one kernel launch (6 frames).
        dyn, stick, refresh = make(n, "cuda", dtype)
        attach_stick(dyn)
        kern = rate(lambda: dyn.step(stick, SUB), n, torch.cuda.synchronize, seconds, refresh)
        ok = ok and bool(torch.isfinite(dyn.rb.pos_ned).all())
        print(f"  GPU {n:>8,} aircraft   torch eager {eager / 1e6:>8.2f} M   torch+graph "
              f"{graphed / 1e6:>8.2f} M   CUDA kernel {kern / 1e6:>9.1f} M"
              f"{'' if ok else '   (NaN!)'}", flush=True)
        cfg = f"{n:,} aircraft"
        rows += [("torch GPU eager", cfg, eager), ("torch GPU + CUDA graph", cfg, graphed),
                 ("CUDA kernel", cfg, kern)]
        del dyn
        torch.cuda.empty_cache()
    return rows


def bench_cpu(sizes, dtypes, threads, seconds) -> list:
    rows = []
    for dt_name in dtypes:
        dtype = dict(float32=torch.float32, float64=torch.float64)[dt_name]
        for th in threads:
            torch.set_num_threads(th)
            for n in sizes:
                dyn, stick, refresh = make(n, "cpu", dtype)
                r = rate(lambda: dyn.step(stick, SUB), n, lambda: None, seconds, refresh,
                         warm=1, every=1)
                print(f"  CPU {dt_name}  {th:>2} thread(s)  {n:>8,} aircraft  {r / 1e6:>9.4f} M",
                      flush=True)
                rows.append((f"torch CPU {dt_name}, {th} thread{'s' if th > 1 else ''}",
                             f"{n:,} aircraft", r))
    return rows


# ---------------------------------------------------------------- JSBSim

def _jsb_new():
    from jsbsim_f16_cuda.f16_reference import JSBSimF16Ref
    ref = JSBSimF16Ref()
    ref.reset(0.0, 0.0, 0.0, v_kt=450.0, h_ft=20000.0)
    fdm = ref.fdm
    trim = (fdm["fcs/aileron-cmd-norm"], -fdm["fcs/elevator-cmd-norm"],
            -fdm["fcs/rudder-cmd-norm"], fdm["fcs/throttle-cmd-norm"])
    return ref, trim


def _jsb_loop(ref, trim, seconds: float, io: bool) -> float:
    """Run frames for `seconds` of busy time and return frames/s (re-trim time excluded)."""
    fdm = ref.fdm
    frames, busy, k = 0, 0.0, 0
    while busy < seconds:
        t0 = time.perf_counter()
        if io:
            for _ in range(1000):
                ref.set_controls(*trim)
                fdm.run()
                for p in READS:
                    fdm[p]
        else:
            for _ in range(1000):
                fdm.run()
        busy += time.perf_counter() - t0
        frames += 1000
        k += 1000
        if k >= RETRIM_FRAMES:
            ref.reset(0.0, 0.0, 0.0, v_kt=450.0, h_ft=20000.0)
            k = 0
    return frames / busy


def _jsb_worker(barrier, out, seconds: float, io: bool) -> None:
    """One process flying one JSBSim aircraft.  The result goes back through the queue only,
    so stdout/stderr are closed (FGFDMExec prints a banner from C++)."""
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    sys.path.insert(0, ROOT)
    ref, trim = _jsb_new()
    _jsb_loop(ref, trim, 0.3, io)                       # warm-up
    barrier.wait()
    out.put(_jsb_loop(ref, trim, seconds, io))


def bench_jsbsim(procs, seconds) -> list:
    import jsbsim
    rows = []
    for io in (False, True):
        tag = "with I/O" if io else "run only"
        for p in procs:
            # Even p = 1 runs in its own process, so every JSBSim number has the same conditions.
            ctx = mp.get_context("spawn")
            barrier, out = ctx.Barrier(p), ctx.Queue()
            ps = [ctx.Process(target=_jsb_worker, args=(barrier, out, seconds, io)) for _ in range(p)]
            for q in ps:
                q.start()
            total = sum(out.get() for _ in ps)
            for q in ps:
                q.join()
            print(f"  JSBSim {jsbsim.__version__}  {tag:<8}  {p:>2} process(es)  {total / 1e6:>9.4f} M "
                  f"({total / p / 1e3:,.1f} k per process)", flush=True)
            rows.append((f"JSBSim {jsbsim.__version__} ({tag})",
                         f"{p} process{'es' if p > 1 else ''}", total))
    return rows


# ---------------------------------------------------------------- summary

def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.machine()


def summary(rows) -> None:
    base = next((r for b, c, r in rows if b.startswith("JSBSim") and "run only" in b
                 and c == "1 process"), None)
    print("\n| backend | config | aircraft-frames/s | vs JSBSim, 1 core |")
    print("|---|---|---|---|")
    for b, c, r in rows:
        x = f"{r / base:,.2f} x" if base else "-"
        v = f"{r / 1e6:,.2f} M" if r >= 1e5 else f"{r / 1e3:,.1f} k"
        print(f"| {b} | {c} | {v} | {x} |")
    if base:
        print(f"\nbaseline: JSBSim run only, 1 process = {base / 1e3:,.1f} k frames/s "
              f"({base / 120:,.0f} x real time)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536, 131072, 262144], help="GPU batch sizes")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"), help="GPU dtype")
    ap.add_argument("--cpu", action="store_true", help="also measure the torch CPU backend")
    ap.add_argument("--cpu-sizes", type=int, nargs="+", default=[1, 64, 1024, 16384])
    ap.add_argument("--cpu-dtypes", nargs="+", default=["float32", "float64"])
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="torch CPU threads (default: 1 and the torch default)")
    ap.add_argument("--jsbsim", action="store_true", help="also measure JSBSim (pip install jsbsim==1.3.0)")
    ap.add_argument("--procs", type=int, nargs="+", default=None,
                    help="JSBSim process counts (default: 1, cores/2, cores, logical cores)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--all", action="store_true", help="GPU + CPU + JSBSim + summary")
    ap.add_argument("--seconds", type=float, default=3.0, help="seconds per measurement")
    ap.add_argument("--json", default="", help="save results as JSON (read by examples/plot_bench.py)")
    a = ap.parse_args()
    if a.all:
        a.cpu = a.jsbsim = True
    logical = os.cpu_count() or 1
    physical = torch.get_num_threads()           # torch default = physical cores
    print(f"CPU {_cpu_name()} ({logical} logical cores), torch {torch.__version__} "
          f"(default {physical} threads), "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    rows = []
    if a.jsbsim:
        procs = a.procs or sorted({1, max(1, physical // 2), physical, logical})
        rows += bench_jsbsim(procs, a.seconds)
    if a.cpu or not torch.cuda.is_available():
        threads = a.threads or sorted({1, physical})
        rows += bench_cpu(a.cpu_sizes, a.cpu_dtypes, threads, a.seconds)
        torch.set_num_threads(physical)
    if torch.cuda.is_available() and not a.no_gpu:
        dtype = dict(float32=torch.float32, float64=torch.float64)[a.dtype]
        rows += bench_gpu(a.sizes, dtype, a.seconds)
    summary(rows)
    if a.json:
        import json
        import jsbsim_f16_cuda
        meta = dict(cpu=_cpu_name(), logical_cores=logical, torch_threads=physical,
                    torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    version=jsbsim_f16_cuda.__version__, seconds=a.seconds,
                    substeps=SUB, date=time.strftime("%Y-%m-%d"))
        try:
            import jsbsim
            meta["jsbsim"] = jsbsim.__version__
        except ImportError:
            pass
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(meta=meta, rows=[dict(backend=b, config=c, rate=r) for b, c, r in rows]),
                      f, indent=1)
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
