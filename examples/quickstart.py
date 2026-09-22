# SPDX-License-Identifier: GPL-3.0-or-later
"""F-16 4,096 대를 무작위 조종간으로 20 초 날린다.

    python examples/quickstart.py            # GPU 가 있으면 cuda, 없으면 cpu
    python examples/quickstart.py --n 256 --device cpu
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, n = a.device, a.n
    g = torch.Generator(device=dev).manual_seed(0)
    rand = lambda *s: torch.rand(*s, device=dev, generator=g)

    dyn = F16Stick(n, device=dev)                    # 위도 0, 기본 연료 3,000 lb, 급유 없음
    alt_ft = 5000.0 + 30000.0 * rand(n)
    pos_ned = torch.stack((torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                           -alt_ft * FT), -1)
    psi = 2.0 * math.pi * rand(n)
    vt_ms = (300.0 + 300.0 * rand(n)) * KT
    ok = dyn.reset(pos_ned, psi, vt_ms)             # 수평 트림 (JSBSim do_trim(1) 자리)
    print(f"{n:,} 대 리셋 (트림 표 안 {ok.float().mean().item() * 100:.1f} %), 장치 {dev}")

    frames = int(a.seconds * 120)
    t0 = time.perf_counter()
    for k in range(frames):
        if k % 120 == 0:                             # 1 초마다 새 조종간
            stick = torch.cat((rand(n, 3) * 1.2 - 0.6, 0.3 + 0.7 * rand(n, 1)), -1)
        dyn.step(stick)                              # = JSBSim set_controls(stick); run()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    s = dyn.state()
    alt1 = -s["pos_ned"][:, 2] / FT
    bank = s["euler"][:, 0].abs() * 180.0 / math.pi
    spd = torch.linalg.vector_norm(s["uvw"], dim=-1) / KT
    print(f"{a.seconds:.0f} 초 = {frames:,} 프레임, {dt:.1f} 초 걸림 "
          f"({n * frames / dt / 1e6:.2f} M 기체·프레임/초, 그래프 없이)")
    print(f"고도 변화 중앙 {(alt1 - alt_ft).median().item():+,.0f} ft, "
          f"속도 중앙 {spd.median().item():.0f} kt, 뱅크 중앙 {bank.median().item():.0f} deg, "
          f"연료 중앙 {s['fuel_lbs'].sum(-1).median().item():,.0f} lb, "
          f"유한 {bool(torch.isfinite(s['pos_ned']).all())}")


if __name__ == "__main__":
    main()
