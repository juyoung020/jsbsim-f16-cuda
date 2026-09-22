# SPDX-License-Identifier: GPL-3.0-or-later
"""처리량 측정: 배치 크기별 기체·프레임/초, eager 대 CUDA 그래프.

    python examples/bench.py                       # 256 ~ 65,536 대
    python examples/bench.py --sizes 4096 16384 --dtype float64

한 번의 `step(stick, substeps=6)` 이 물리 6 프레임(1/20 s)이다.  CUDA 그래프는 그
한 호출을 통째로 캡처해서 재생한다 (상태는 전부 제자리 갱신이라 그대로 된다).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jsbsim_f16_cuda import F16Stick  # noqa: E402

FT, KT = 0.3048, 0.514444
SUB = 6


def make(n: int, dev: str, dtype) -> tuple:
    dyn = F16Stick(n, device=dev, dtype=dtype)
    g = torch.Generator(device=dev).manual_seed(0)
    z = torch.zeros(n, device=dev, dtype=dtype)
    alt = (10000.0 + 20000.0 * torch.rand(n, device=dev, generator=g, dtype=dtype)) * FT
    dyn.reset(torch.stack((z, z, -alt), -1), 2 * math.pi * torch.rand(n, device=dev, generator=g, dtype=dtype),
              (350.0 + 200.0 * torch.rand(n, device=dev, generator=g, dtype=dtype)) * KT)
    stick = torch.tensor([0.1, 0.3, 0.0, 0.8], device=dev, dtype=dtype).expand(n, 4).clone()
    return dyn, stick


def rate(fn, n: int, sync, seconds: float = 2.0) -> float:
    for _ in range(3):
        fn()
    sync()
    calls, t0 = 0, time.perf_counter()
    while True:
        fn()
        calls += 1
        if calls % 10 == 0:
            sync()
            dt = time.perf_counter() - t0
            if dt > seconds:
                return n * SUB * calls / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096, 16384, 65536])
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--cpu", action="store_true", help="CPU 도 잰다 (작은 배치만)")
    a = ap.parse_args()
    dtype = dict(float32=torch.float32, float64=torch.float64)[a.dtype]
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 없음'}, {a.dtype}")
    print(f"{'기체 수':>8} | {'eager':>14} | {'CUDA 그래프':>14} | {'실시간 배수 (그래프)':>18}")
    print(f"{'':>8} | {'M 기체·프레임/초':>14} | {'M 기체·프레임/초':>14} |")
    for n in a.sizes:
        if not torch.cuda.is_available():
            break
        dyn, stick = make(n, "cuda", dtype)
        eager = rate(lambda: dyn.step(stick, SUB), n, torch.cuda.synchronize)
        # CUDA 그래프: 옆 스트림에서 몸을 푼 뒤 한 호출을 캡처한다.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                dyn.step(stick, SUB)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dyn.step(stick, SUB)
        graphed = rate(graph.replay, n, torch.cuda.synchronize)
        ok = bool(torch.isfinite(dyn.rb.pos_ned).all())
        print(f"{n:>8,} | {eager / 1e6:>14.2f} | {graphed / 1e6:>14.2f} | "
              f"{graphed / n / 120:>16,.0f} x{'' if ok else '   (NaN!)'}")
        del dyn, graph
        torch.cuda.empty_cache()
    if a.cpu or not torch.cuda.is_available():
        for n in (64, 256, 1024):
            dyn, stick = make(n, "cpu", dtype)
            r = rate(lambda: dyn.step(stick, SUB), n, lambda: None)
            print(f"{n:>8,} | {r / 1e6:>14.3f} | {'(CPU)':>14} |")


if __name__ == "__main__":
    main()
