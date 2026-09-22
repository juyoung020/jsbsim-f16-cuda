# SPDX-License-Identifier: GPL-3.0-or-later
"""TankMass 대 JSBSim 점검, CUDA 커널 대 torch 점검.

    python -m jsbsim_f16_cuda.f16_check mass
    python -m jsbsim_f16_cuda.f16_check fused     # 커널 대 torch F16Stick (float64 / float32)

`mass`:

  1. 정적: 탱크 네 개에 무작위 연료를 넣고 `run_ic()` 한 뒤 JSBSim 의 weight, cg,
     ixx/iyy/izz/ixz 와 `TankMass.props` 를 비교한다.
  2. 동적: 무작위 스로틀로 날리며 매 프레임 JSBSim 이 쓴 연료유량으로
     `TankMass.advance` 를 한 걸음씩 밟아 탱크별 연료를 비교하고, 그 프레임
     JSBSim 이 보고한 질량·관성이 **직전 프레임 연료**의 `props` 와 같은지 본다
     (`FGMassBalance` 가 `FGPropulsion` 보다 먼저 돌고, 탱크 평행축 항은 그보다
     한 프레임 전 무게중심 기준이다).  급유 켬/끔, 외부탱크 포함/제외, 탱크가
     바닥나는 경우를 다 돈다.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from jsbsim_f16_cuda.f16_reference import JSBSimF16Ref  # noqa: E402

INERTIA = ("ixx", "iyy", "izz", "ixz")


def _jsb_mass(fdm) -> dict:
    return dict(weight=fdm["inertia/weight-lbs"],
                cg=np.array([fdm["inertia/cg-x-in"], fdm["inertia/cg-y-in"],
                             fdm["inertia/cg-z-in"]]),
                **{k: fdm[f"inertia/{k}-slugs_ft2"] for k in INERTIA})


def _ours(tm, fuel, cg_tank=None):
    import torch
    from jsbsim_f16_cuda.f16_core import SLUGFT2
    mp, cg, m_sl = tm.props(torch.tensor(np.array([fuel]), dtype=torch.float64), cg_tank)
    return dict(weight=float(m_sl[0]) * 32.174049, cg=cg[0].numpy(),
                **{k: float(getattr(mp, k)[0]) / SLUGFT2 for k in INERTIA})


def _diff(a: dict, b: dict) -> dict:
    return {k: float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in a}


def check_mass(n_static: int, frames: int, seed: int) -> int:
    import torch
    from jsbsim_f16_cuda.f16_core import TankMass
    rng = np.random.default_rng(seed)
    cap = np.array(TankMass.TANK_CAP_LBS)
    tm = TankMass(1, "cpu", torch.float64)
    ref = JSBSimF16Ref()
    fdm = ref.fdm
    worst = {}
    for i in range(n_static):
        fuel = rng.uniform(0.0, 1.0, 4) * cap * (rng.uniform() < 0.8)
        for t, c in enumerate(fuel):
            fdm[f"propulsion/tank[{t}]/contents-lbs"] = float(c)
        fdm.run_ic()
        for k, v in _diff(_jsb_mass(fdm), _ours(tm, fuel)).items():
            worst[k] = max(worst.get(k, 0.0), v)
    print(f"정적 (무작위 탱크 배치 {n_static} 개): 최대 |TankMass - JSBSim|")
    print("  " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))

    cases = (("기본 연료, 급유 끔", False, TankMass.DEFAULT_LBS),
             ("만탱크, 급유 끔", False, TankMass.TANK_CAP_LBS),
             ("기본 연료, 급유 켬", True, TankMass.DEFAULT_LBS),
             ("거의 빈 탱크 (바닥남)", False, (40.0, 12.0, 0.0, 0.0)),
             ("외부만, 급유 켬", True, (0.0, 0.0, 2000.0, 2990.5)))
    bad = 0
    print(f"동적 ({frames} 프레임 = {frames / 120:.0f} 초, 무작위 스로틀): 최대 |오차|")
    for name, refuel, f0 in cases:
        r = JSBSimF16Ref(refuel=refuel)
        r.reset(0.0, 0.0, 0.0, v_kt=450.0, h_ft=20000.0, fuel_lbs=f0)
        tmr = TankMass(1, "cpu", torch.float64, refuel=refuel)
        prev = np.array(r.tanks())
        cg_prev = None                     # 트림 직후: 탱크 항 기준 = 지금 무게중심
        w_tank, w_mass = 0.0, {}
        thr = 1.0
        for k in range(frames):
            if k % 90 == 0:
                thr = float(rng.uniform(0.0, 1.0))
            r.set_controls(0.0, 0.0, 0.0, thr)
            r.run_one()
            now = np.array(r.tanks())
            ff = r.fdm["propulsion/engine[0]/fuel-flow-rate-pps"]
            ours = tmr.advance(torch.tensor(np.array([prev]), dtype=torch.float64),
                               torch.tensor([ff], dtype=torch.float64))[0].numpy()
            w_tank = max(w_tank, float(np.abs(ours - now).max()))
            got = _ours(tmr, prev, cg_prev)
            for kk, v in _diff(_jsb_mass(r.fdm), got).items():
                w_mass[kk] = max(w_mass.get(kk, 0.0), v)
            cg_prev = torch.tensor(np.array([got["cg"]]), dtype=torch.float64)
            prev = now
        flag = w_tank > 1e-9 or max(w_mass.values()) > 1e-6
        bad += flag
        print(f"  {name:<22} 탱크 {w_tank:.1e} lb   " + "  ".join(
            f"{k} {v:.1e}" for k, v in w_mass.items()) + ("   <-- 어긋남" if flag else "")
              + f"   (끝 연료 {now.sum():,.1f} lb)")
    print("통과" if not bad else f"실패 {bad} 건")
    return 1 if bad else 0


def check_fused(n: int, steps: int, seed: int) -> int:
    """CUDA 커널(`fused_core.attach_stick`) 대 torch F16Stick.

    두 가지로 잰다.
      매 스텝 맞춤  스텝마다 커널 쪽 상태를 torch 상태로 덮어쓴 뒤 같은 조종으로 1~6 프레임.
                    누적이 없으니 남는 차이는 식의 차이다.
      자유 비행     처음에만 맞추고 `steps` 스텝을 따로 날린다 (float64 에서 반올림이 쌓이는 정도).
    구성: 탱크 질량 급유 끔 (위도 0), 급유 켬 (위도 60).  float64 와 float32.
    """
    import torch
    from jsbsim_f16_cuda.f16_core import F16Stick
    from jsbsim_f16_cuda import fused_core as FK
    if not torch.cuda.is_available():
        print("CUDA 가 없어 건너뛴다")
        return 0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bad = 0
    print(f"CUDA 커널 대 torch F16Stick ({n} 대, 스텝마다 1~6 프레임 무작위 조종, 상대오차 = "
          f"|차| / 그 상태의 최대 |값|)")
    for dt in (torch.float64, torch.float32):
        for tag, kw in (("급유 끔, 위도 0", dict(lat0_deg=0.0)),
                        ("급유 켬, 위도 60", dict(lat0_deg=60.0, refuel=True))):
            ta, tb = F16Stick(n, "cuda", dt, **kw), F16Stick(n, "cuda", dt, **kw)
            pos = torch.zeros(n, 3, device="cuda", dtype=dt)
            pos[:, 2] = -(1500.0 + 9000.0 * torch.rand(n, device="cuda", dtype=dt, generator=gen))
            psi = torch.rand(n, device="cuda", dtype=dt, generator=gen) * 6.2 - 3.1
            vt = 130.0 + 180.0 * torch.rand(n, device="cuda", dtype=dt, generator=gen)
            fuel = torch.rand(n, 4, device="cuda", dtype=dt, generator=gen) * torch.tensor(
                [3486.0, 3486.0, 2991.0, 2991.0], device="cuda", dtype=dt)
            for d in (ta, tb):
                d.reset(pos, psi, vt, fuel)
            FK.attach_stick(tb)

            def sync():
                for k, t in tb.watched_tensors().items():
                    if t.numel():
                        t.copy_(ta.watched_tensors()[k])

            def diff():
                w = 0.0
                for k, x in ta.watched_tensors().items():
                    y = tb.watched_tensors()[k]
                    if not x.numel() or x.dtype == torch.bool:
                        if x.numel() and bool((x != y).any()):
                            return float("inf")
                        continue
                    w = max(w, float((x.double() - y.double()).abs().max()
                                     / (x.double().abs().max() + 1e-12)))
                return w

            def stick():
                u = torch.rand(n, 4, device="cuda", dtype=dt, generator=gen) * 2 - 1
                u[:, 3] = u[:, 3].abs()
                return u

            step_w = 0.0
            for i in range(steps):
                sync()
                u, sub = stick(), 1 + i % 6
                F16Stick.step(ta, u, sub)
                tb.step(u, sub)
                step_w = max(step_w, diff())
            ta.reset(pos, psi, vt, fuel)
            sync()
            for i in range(steps):
                u, sub = stick(), 1 + i % 6
                F16Stick.step(ta, u, sub)
                tb.step(u, sub)
            free_w = diff()
            lim = 1e-11 if dt == torch.float64 else 1e-3   # float32: 반올림 + 포화 분기
            flag = step_w > lim
            bad += flag
            name = "float64" if dt == torch.float64 else "float32"
            print(f"  {name}  {tag:<14} 매 스텝 맞춤 최대 {step_w:.1e}   "
                  f"자유 비행 {steps} 스텝 뒤 {free_w:.1e}{'   <-- 임계 초과' if flag else ''}")
    print("통과" if not bad else f"실패 {bad} 건")
    return 1 if bad else 0




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("mass", "fused", "reference"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--frames", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.what == "mass":
        return check_mass(a.n, a.frames, a.seed)
    if a.what == "fused":
        return check_fused(a.n if a.n != 200 else 512, 60, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
