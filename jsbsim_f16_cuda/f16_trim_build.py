# SPDX-License-Identifier: GPL-3.0-or-later
"""수평 트림 표 `f16_trim.npz` 를 JSBSim 에서 잰다 (`F16Stick.reset` 이 쓴다).

    python f16_trim_build.py            # 표를 만든다 (JSBSim do_trim 8,778 회, 1 분 남짓)
    python f16_trim_build.py --check    # 격자 밖 무작위 점에서 표 대 JSBSim 을 대조한다

격자: 해면고도 0 ~ 45,000 ft (2,500 ft 간격) x 진대기속도 150 ~ 800 kt (10 kt)
x 연료 합계 {500, 1,500, 3,000, 4,500, 6,972, 9,000, 12,954} lb (내부 탱크 먼저, `TankMass.internal_first`).
각 점에서 JSBSim 표준 설정(`JSBSimF16Ref()`, 위도 0)으로 `do_trim(1)` 을 돌려

    theta, phi                  트림 자세 [rad]
    pitch_trim                  fcs/pitch-trim-cmd-norm
    elevator_rad                fcs/elevator-pos-rad (FLCS 지연버퍼 씨앗)
    throttle                    fcs/throttle-cmd-norm (트림 프레임에 들고 있는 스로틀)
    n2                          propulsion/engine[0]/n2 [%]
    ok                          do_trim 수렴 여부

를 적는다.  속도 격자가 10 kt 인 이유: 받음각이 대략 1/V^2 이라 볼록해서, 50 kt
격자면 선형보간이 칸 가운데서 0.1 도 가까이 빗나간다.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from jsbsim_f16_cuda.f16_reference import JSBSimF16Ref  # noqa: E402

H_FT = tuple(float(h) for h in range(0, 45001, 2500))
V_KT = tuple(float(v) for v in range(150, 801, 10))
FUEL_LBS = (500.0, 1500.0, 3000.0, 4500.0, 6972.0, 9000.0, 12954.0)
OUT = os.path.join(HERE, "f16_trim.npz")
PROPS = dict(theta="attitude/theta-rad", phi="attitude/phi-rad",
             pitch_trim="fcs/pitch-trim-cmd-norm", elevator_rad="fcs/elevator-pos-rad",
             throttle="fcs/throttle-cmd-norm", n2="propulsion/engine[0]/n2")


def tanks(total: float) -> tuple:
    cap_in = 3486.0 * 2
    t = min(max(total, 0.0), 12954.0)
    inner = min(t, cap_in) / 2.0
    outer = max(t - cap_in, 0.0) / 2.0
    return (inner, inner, outer, outer)


def trim_point(ref: JSBSimF16Ref, h: float, v: float, fuel: float) -> tuple:
    ok = ref.reset(0.0, 0.0, 0.0, v_kt=v, h_ft=h, fuel_lbs=tanks(fuel))
    return ok, {k: float(ref.fdm[p]) for k, p in PROPS.items()}


def build() -> int:
    ref = JSBSimF16Ref()
    shape = (len(H_FT), len(V_KT), len(FUEL_LBS))
    ok = np.zeros(shape, dtype=bool)
    vals = {k: np.full(shape, np.nan) for k in PROPS}
    t0 = time.time()
    for i, h in enumerate(H_FT):
        for j, v in enumerate(V_KT):
            for k, f in enumerate(FUEL_LBS):
                good, row = trim_point(ref, h, v, f)
                ok[i, j, k] = good
                for key, x in row.items():
                    vals[key][i, j, k] = x
        print(f"  h {h:>7.0f} ft  트림 {ok[i].mean() * 100:5.1f} %", flush=True)
    np.savez(OUT, h_ft=np.array(H_FT), v_kt=np.array(V_KT),
             fuel_lbs=np.array(FUEL_LBS), ok=ok, **vals)
    print(f"{OUT}: {ok.size:,} 점 중 트림 {ok.sum():,} ({ok.mean() * 100:.1f} %), "
          f"{time.time() - t0:.0f} 초")
    return 0


def check(n: int, seed: int, hold_s: float) -> int:
    """표 보간 대 JSBSim do_trim (정적), 그리고 트림 조종을 들고 `hold_s` 초 (동적)."""
    import torch
    from jsbsim_f16_cuda.f16_core import F16Stick, TrimGrid, KT_TO_MS, FT
    grid = TrimGrid(OUT, dtype=torch.float64)
    rng = np.random.default_rng(seed)
    ref = JSBSimF16Ref()
    pts, stat = [], {k: [] for k in PROPS}
    while len(pts) < n:
        h, v = rng.uniform(H_FT[0], H_FT[-1]), rng.uniform(V_KT[0], V_KT[-1])
        f = rng.uniform(FUEL_LBS[0], FUEL_LBS[-1])
        tr, okg = grid(torch.tensor([h], dtype=torch.float64), torch.tensor([v], dtype=torch.float64),
                       torch.tensor([f], dtype=torch.float64))
        if not bool(okg[0]):
            continue
        good, row = trim_point(ref, h, v, f)
        if not good:
            continue
        pts.append((h, v, f, row))
        for k in PROPS:
            stat[k].append(abs(float(tr[k][0]) - row[k]))
    print(f"정적: 격자 밖 무작위 {n} 점 (표가 트림 가능이라 한 곳), |보간 - JSBSim|")
    for k, e in stat.items():
        e = np.array(e)
        print(f"  {k:<13} 중앙 {np.median(e):.2e}   p99 {np.quantile(e, 0.99):.2e}   최대 {e.max():.2e}")

    # 동적: 같은 점에서 시작해 트림 조종을 그대로 들고 난다.
    frames = int(round(hold_s * 120))
    m = len(pts)
    dyn = F16Stick(m, device="cpu", dtype=torch.float64)
    pos = torch.tensor([[0.0, 0.0, -h * FT] for h, _, _, _ in pts], dtype=torch.float64)
    vt = torch.tensor([v * KT_TO_MS for _, v, _, _ in pts], dtype=torch.float64)
    fuel = torch.tensor([tanks(f) for _, _, f, _ in pts], dtype=torch.float64)
    dyn.reset(pos, torch.zeros(m, dtype=torch.float64), vt, fuel)
    stick = dyn.stick.clone()
    for _ in range(frames):
        dyn.step(stick)
    s = dyn.state()
    dh_g = (-s["pos_ned"][:, 2] / FT).numpy() - np.array([h for h, _, _, _ in pts])
    dv_g = (torch.linalg.vector_norm(s["uvw"], dim=-1) / KT_TO_MS).numpy() - np.array([v for _, v, _, _ in pts])
    dh_j, dv_j = [], []
    for h, v, f, row in pts:
        trim_point(ref, h, v, f)
        ref.set_controls(0.0, 0.0, 0.0, row["throttle"])
        for _ in range(frames):
            ref.run_one()
        dh_j.append(ref.fdm["position/h-sl-ft"] - h)
        dv_j.append(ref.fdm["velocities/vtrue-kts"] - v)
    dh_j, dv_j = np.array(dh_j), np.array(dv_j)
    print(f"동적: 트림 조종을 {hold_s:.0f} 초 들고 비행 (같은 {m} 점)")
    for name, g, j, unit in (("고도 변화", dh_g, dh_j, "ft"), ("속도 변화", dv_g, dv_j, "kt")):
        d = np.abs(g - j)
        w = int(d.argmax())
        print(f"    최악 h {pts[w][0]:.0f} ft  v {pts[w][1]:.0f} kt  연료 {pts[w][2]:.0f} lb: "
              f"GPU {g[w]:+.2f}  JSBSim {j[w]:+.2f} {unit}")
        print(f"  {name}  GPU 중앙 |{np.median(np.abs(g)):.3f}|  JSBSim 중앙 |{np.median(np.abs(j)):.3f}|"
              f"   |GPU-JSBSim| 중앙 {np.median(d):.3f}  최대 {d.max():.3f} {unit}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=float, default=10.0)
    a = ap.parse_args()
    return check(a.n, a.seed, a.hold) if a.check else build()


if __name__ == "__main__":
    raise SystemExit(main())
