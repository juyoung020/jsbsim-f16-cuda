# SPDX-License-Identifier: GPL-3.0-or-later
"""Fly 65,536 F-16s for 20 s with random stick inputs.

    python examples/quickstart.py                      # CUDA kernel
    python examples/quickstart.py --backend torch      # torch reference backend on the GPU
    python examples/quickstart.py --device cpu --n 1024
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jsbsim_f16_cuda import F16Stick, attach_stick  # noqa: E402

FT, KT = 0.3048, 0.514444


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="kernel", choices=("kernel", "torch"))
    a = ap.parse_args()
    dev, n = a.device, a.n
    g = torch.Generator(device=dev).manual_seed(0)
    rand = lambda *s: torch.rand(*s, device=dev, generator=g)

    dyn = F16Stick(n, device=dev)                    # latitude 0, default fuel 3,000 lb, no refuel
    if a.backend == "kernel" and dev == "cuda":
        attach_stick(dyn)                            # dyn.step now launches the CUDA kernel
    alt_ft = 5000.0 + 30000.0 * rand(n)
    pos_ned = torch.stack((torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                           -alt_ft * FT), -1)       # north, east, down [m]
    psi = 2.0 * math.pi * rand(n)                    # true heading [rad]
    vt_ms = (300.0 + 300.0 * rand(n)) * KT           # true airspeed [m/s]
    ok = dyn.reset(pos_ned, psi, vt_ms)             # level trim, like JSBSim do_trim(1)
    print(f"{n:,} aircraft reset ({ok.float().mean().item() * 100:.1f} % inside the trim table), "
          f"device {dev}, backend {a.backend if dev == 'cuda' else 'torch'}")

    frames = int(a.seconds * 120)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(0, frames, 6):
        if k % 120 == 0:                             # new random stick every second
            stick = torch.cat((rand(n, 3) * 1.2 - 0.6, 0.3 + 0.7 * rand(n, 1)), -1)
        dyn.step(stick, substeps=6)                  # 6 x (JSBSim set_controls(); run())
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    s = dyn.state()
    alt1 = -s["pos_ned"][:, 2] / FT
    bank = s["euler"][:, 0].abs() * 180.0 / math.pi
    spd = torch.linalg.vector_norm(s["uvw"], dim=-1) / KT
    print(f"{a.seconds:.0f} s of flight = {frames:,} frames per aircraft in {dt:.2f} s wall time "
          f"({n * frames / dt / 1e6:,.1f} M aircraft-frames/s, no CUDA graph)")
    print(f"median altitude change {(alt1 - alt_ft).median().item():+,.0f} ft, "
          f"speed {spd.median().item():.0f} kt, |bank| {bank.median().item():.0f} deg, "
          f"fuel {s['fuel_lbs'].sum(-1).median().item():,.0f} lb, "
          f"all finite: {bool(torch.isfinite(s['pos_ned']).all())}")


if __name__ == "__main__":
    main()
