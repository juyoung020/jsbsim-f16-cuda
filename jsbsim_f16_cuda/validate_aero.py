# SPDX-License-Identifier: GPL-3.0-or-later
"""JSBSim 을 진실값으로 삼아 aero.py / propulsion.py 를 대조한다.

    python jsbsim_f16_cuda/validate_aero.py            # 기본 2,000 표본
    python jsbsim_f16_cuda/validate_aero.py -n 8000 --device cuda

방법
----
JSBSim 판을 하나 띄우고 포락선 곳곳에 상태를 뿌린 뒤, **그 프레임에 JSBSim 이
실제로 쓴 입력**(aero/alpha-rad, aero/qbar-psf, fcs/*-pos-rad, ...)을 프로퍼티로
읽어 우리 구현에 그대로 넣는다.  입력을 우리가 다시 계산하지 않고 읽어 오는
것이 요점이다 -- 그래야 오차가 **보간 구현의 오차만** 남는다.  대기·FCS 가
섞이면 어느 쪽이 틀렸는지 알 수 없다.

JSBSim 한 프레임의 모델 순서는 Propagate -> Atmosphere -> Auxiliary -> FCS ->
Aerodynamics -> Accelerations 라, run() 이 돌아온 뒤 읽는 alpha/qbar/조종면과
aero/coefficient/* 는 같은 순간의 짝이다.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jsbsim  # noqa: E402

from jsbsim_f16_cuda.aero import AXES, AeroState, F16Aero  # noqa: E402
from jsbsim_f16_cuda.propulsion import F100Turbine, StdAtmosphere, _RHO_SL  # noqa: E402

INPUTS = {
    "alpha": "aero/alpha-rad", "beta": "aero/beta-rad", "mach": "velocities/mach",
    "qbar": "aero/qbar-psf", "vt": "velocities/vtrue-fps",
    "p": "velocities/p-aero-rad_sec", "q": "velocities/q-aero-rad_sec",
    "r": "velocities/r-aero-rad_sec",
    "elevator": "fcs/elevator-pos-rad", "aileron": "fcs/aileron-pos-rad",
    "rudder": "fcs/rudder-pos-rad", "lef": "fcs/lef-pos-rad",
    "speedbrake": "fcs/speedbrake-pos-rad", "flaperon_mix": "fcs/flaperon-mix-rad",
    "gear": "gear/gear-pos-norm", "h_b_mac": "aero/h_b-mac-ft",
}
EXTRA = {
    "bi2vel": "aero/bi2vel", "ci2vel": "aero/ci2vel", "kCLge": "aero/function/kCLge",
    "fbx": "forces/fbx-aero-lbs", "fby": "forces/fby-aero-lbs",
    "fbz": "forces/fbz-aero-lbs", "fwx": "forces/fwx-aero-lbs",
    "fwy": "forces/fwy-aero-lbs", "fwz": "forces/fwz-aero-lbs",
    "Ml": "moments/l-aero-lbsft", "Mm": "moments/m-aero-lbsft",
    "Mn": "moments/n-aero-lbsft",
    "cgx": "inertia/cg-x-in", "cgy": "inertia/cg-y-in", "cgz": "inertia/cg-z-in",
    "h": "position/h-sl-ft", "rho": "atmosphere/rho-slugs_ft3",
    "P": "atmosphere/P-psf", "T": "atmosphere/T-R", "a": "atmosphere/a-fps",
    "dalt": "atmosphere/density-altitude",
    "thrust": "propulsion/engine[0]/thrust-lbs",
    "n2": "propulsion/engine[0]/n2", "thr_pos": "fcs/throttle-pos-norm",
}


def new_fdm():
    root = os.path.dirname(jsbsim.__file__)
    fdm = jsbsim.FGFDMExec(root, None)
    fdm.set_debug_level(0)
    fdm.load_model("f16")
    fdm.set_dt(1.0 / 120.0)
    return fdm


def _condition(fdm, rng, frames: int = 50):
    """얌전한 조건에서 프레임을 굴려 조종면을 현재 명령 쪽으로 몰아 둔다.

    액추에이터는 0.3~1.0 s 짜리 레이트 제한기라 한 프레임으로는 안 움직인다.
    포락선 표본은 프레임을 1 번만 돌리므로(고 alpha 에서 여러 프레임을 돌리면
    텀블해 NaN 이 난다) 조종면 커버리지는 여기서 벌어 둬야 한다.
    """
    fdm["ic/h-sl-ft"] = 20000.0
    for k in ("phi-deg", "theta-deg", "alpha-deg", "beta-deg", "gamma-deg",
              "p-rad_sec", "q-rad_sec", "r-rad_sec"):
        fdm["ic/" + k] = 0.0
    fdm["ic/mach"] = float(rng.uniform(0.4, 0.9))
    fdm.run_ic()
    fdm["propulsion/engine[0]/set-running"] = 1
    for _ in range(frames):
        fdm.run()


def sample(n: int, seed: int = 0):
    """포락선 전체에 뿌린 표본.  입력·출력 모두 JSBSim 에서 읽는다.

    조종면을 골고루 깔려면 요령이 하나 필요하다.  `run_ic()` 는 리셋이 아니라
    **FCS 상태를 그대로 둔다** (`f16_reference` 함정 2 번).
    그래서 같은 조종 명령을 20 표본쯤 이어서 걸어 두면 액추에이터가 그쪽으로
    램프를 타는 동안, 기체 상태는 표본마다 포락선 아무 데나 다시 던질 수 있다.
    프레임은 표본당 1 번만 돌린다 -- alpha 35 도에서 10 프레임을 돌리면 기체가
    텀블해 NaN 으로 죽고, 그 인스턴스는 그 뒤로 전부 NaN 을 뱉는다.
    """
    rng = np.random.default_rng(seed)
    fdm = new_fdm()
    rec = {k: [] for k in list(INPUTS) + list(EXTRA) + ["fn:" + s for s in FN_NAMES]}

    i = 0
    nan_retry = 0
    while i < n:
        if i % 24 == 0:          # 조종면 목표를 새로 뽑아 그쪽으로 램프시킨다
            fdm["fcs/aileron-cmd-norm"] = rng.uniform(-1, 1)
            fdm["fcs/elevator-cmd-norm"] = rng.uniform(-1, 1)
            fdm["fcs/rudder-cmd-norm"] = rng.uniform(-1, 1)
            fdm["fcs/throttle-cmd-norm"] = rng.uniform(0, 1)
            fdm["fcs/fbw-override"] = float(rng.random() < 0.7)
            fdm["fcs/speedbrake-cmd-norm"] = float(rng.random() < 0.35)
            fdm["gear/gear-cmd-norm"] = float(rng.random() < 0.35)
            _condition(fdm, rng)             # 얌전한 조건에서 액추에이터를 굴린다

        # IC 순서가 중요하다.  각도를 먼저 0 으로 밀고 -> 속도 -> 각도 순으로
        # 넣지 않으면 JSBSim 이 "Cannot modify angle 'alpha'" 로 조용히 무시한다.
        # theta 대신 gamma 를 쓴다 (theta 를 주면 alpha 와 충돌한다).
        fdm["ic/h-sl-ft"] = rng.uniform(5000.0, 30000.0)
        fdm["ic/psi-true-deg"] = rng.uniform(0.0, 360.0)
        for k in ("phi-deg", "theta-deg", "alpha-deg", "beta-deg", "gamma-deg",
                  "p-rad_sec", "q-rad_sec", "r-rad_sec"):
            fdm["ic/" + k] = 0.0
        fdm["ic/mach"] = rng.uniform(0.2, 1.6)
        fdm["ic/alpha-deg"] = rng.uniform(-10.0, 40.0)
        fdm["ic/beta-deg"] = rng.uniform(-20.0, 20.0)
        fdm["ic/gamma-deg"] = rng.uniform(-45.0, 45.0)
        fdm["ic/phi-deg"] = rng.uniform(-180.0, 180.0)
        fdm["ic/p-rad_sec"] = rng.uniform(-3.0, 3.0)
        fdm["ic/q-rad_sec"] = rng.uniform(-1.5, 1.5)
        fdm["ic/r-rad_sec"] = rng.uniform(-1.0, 1.0)
        fdm.run_ic()
        fdm["propulsion/engine[0]/set-running"] = 1
        fdm.run()

        if not math.isfinite(fdm["aero/qbar-psf"]):
            nan_retry += 1
            fdm = new_fdm()                 # 한 번 NaN 이 나면 그 판은 못 쓴다
            _condition(fdm, rng)
            continue
        for k, p in INPUTS.items():
            rec[k].append(fdm[p])
        for k, p in EXTRA.items():
            rec[k].append(fdm[p])
        for s in FN_NAMES:
            rec["fn:" + s].append(fdm[s])
        i += 1
    if nan_retry:
        print(f"  (NaN 으로 버린 표본 {nan_retry} 개 -- 판을 다시 띄웠다)")
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}


def errtable(title, rows):
    """rows: (이름, 참값 (N,), 우리값 (N,))."""
    print(f"\n{title}")
    print(f"  {'quantity':26s} {'|max|':>11s} {'rel max':>10s} {'rel med':>10s} "
          f"{'abs max':>11s}  worst-at")
    out = []
    for name, ref, mine in rows:
        ref = np.asarray(ref, dtype=np.float64)
        mine = np.asarray(mine, dtype=np.float64)
        d = np.abs(mine - ref)
        scale = np.maximum(np.abs(ref), 1e-3 * np.max(np.abs(ref)) + 1e-30)
        rel = d / scale
        k = int(np.argmax(rel))
        out.append((name, np.max(np.abs(ref)), rel.max(), np.median(rel), d.max(), k))
        print(f"  {name:26s} {out[-1][1]:11.4g} {rel.max():10.3e} "
              f"{np.median(rel):10.3e} {d.max():11.4g}  #{k}")
    return out


def bench(device, dtype, sizes=(1024, 4096, 16384), iters=200):
    """계수 계산 한 번에 걸리는 시간.  JSBSim 은 없어도 된다."""
    aero = F16Aero(device=device, dtype=dtype)
    eng = F100Turbine(device=device, dtype=dtype)
    cuda = str(device).startswith("cuda")
    print(f"\n== 속도 ({device}, {str(dtype).replace('torch.','')}) ==")
    print(f"  {'batch':>7s} {'aero us':>10s} {'aero M/s':>10s} "
          f"{'engine us':>10s} {'both us':>9s} {'both M/s':>9s} "
          f"{'graph us':>11s} {'graph M/s':>11s}")
    for B in sizes:
        g = torch.Generator(device="cpu").manual_seed(0)
        rnd = lambda lo, hi: (torch.rand(B, generator=g) * (hi - lo) + lo).to(device, dtype)
        st = AeroState(alpha=rnd(-0.17, 0.7), beta=rnd(-0.35, 0.35), mach=rnd(0.2, 1.6),
                       qbar=rnd(100, 2000), vt=rnd(300, 1600), p=rnd(-3, 3),
                       q=rnd(-1.5, 1.5), r=rnd(-1, 1), elevator=rnd(-.436, .436),
                       aileron=rnd(-.375, .375), rudder=rnd(-.524, .524),
                       lef=rnd(0, .436), speedbrake=rnd(0, 1.05),
                       flaperon_mix=rnd(-1, 1), gear=torch.zeros(B, device=device, dtype=dtype)).fill()
        n2 = rnd(53, 100); n2n = (n2 - 53.0) / 47.0
        cmd = rnd(0, 1); h = rnd(5000, 30000)

        def once():
            aero(st)
            eng.step(n2, n2n, cmd, st.mach, h, 1 / 120.)

        for f, lbl in ((lambda: aero(st), "aero"),
                       (lambda: eng.step(n2, n2n, cmd, st.mach, h, 1 / 120.), "eng"),
                       (once, "both")):
            for _ in range(30):
                f()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                f()
            if cuda:
                torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6
            if lbl == "aero":
                a_us = us
            elif lbl == "eng":
                e_us = us
            else:
                b_us = us
        g_us = float("nan")
        if cuda:
            g_us = _graph_time(once, iters)
        print(f"  {B:7d} {a_us:10.1f} {B/a_us:10.2f} {e_us:10.1f} "
              f"{b_us:9.1f} {B/b_us:9.2f} {g_us:11.1f} {B/g_us:11.2f}")
    if cuda:
        print(f"  테이블 상주 메모리 {sum(b.numel()*b.element_size() for b in aero.buffers())/1024:.1f} KB"
              f" (aero) + {sum(b.numel()*b.element_size() for b in eng.buffers())/1024:.1f} KB (engine)")
        print("  CUDA 그래프가 4 배쯤 벌어 준다 -- 이 부분은 계산량이 아니라"
              " 커널 런치 수에 묶여 있다 (약 90 개).")


def _graph_time(fn, iters):
    """CUDA 그래프로 캡처해 재생 시간을 잰다.  데이터 의존 분기가 없어 캡처된다."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--bench", action="store_true", help="속도만 재고 끝낸다")
    args = ap.parse_args()

    dtype = torch.float64 if args.float64 else torch.float32
    dev = args.device
    if args.bench:
        bench(dev, dtype)
        return
    aero = F16Aero(device=dev, dtype=dtype)
    eng = F100Turbine(device=dev, dtype=dtype)
    atm = StdAtmosphere(device=dev, dtype=dtype)

    print(f"표본 {args.n} 개를 JSBSim 에서 모은다 ...")
    d = sample(args.n)
    T = lambda k: torch.as_tensor(d[k], dtype=dtype, device=dev)

    st = AeroState(**{k: T(k) for k in INPUTS}).fill()
    out = aero(st, want_functions=True)
    fns = out["functions"].double().cpu().numpy()

    # ---------------------------------------------------------------- 0. 입력
    print("\n== 표본 범위 ==")
    for k in ("alpha", "beta", "mach", "qbar", "elevator", "aileron", "rudder",
              "lef", "speedbrake", "flaperon_mix", "gear", "h"):
        v = d[k]
        print(f"  {k:14s} [{v.min():10.4f}, {v.max():10.4f}]  고유값 {len(np.unique(np.round(v,6))):5d}")

    # -------------------------------------------------- 1. 파생량 (bi2vel 등)
    errtable("== 파생 입력 (우리가 계산, JSBSim 이 진실값) ==", [
        ("aero/bi2vel", d["bi2vel"], (30.0 / (2.0 * d["vt"]))),
        ("aero/ci2vel", d["ci2vel"], (11.32 / (2.0 * d["vt"]))),
        ("aero/function/kCLge", d["kCLge"],
         aero.kCLge(T("h_b_mac")).double().cpu().numpy()),
    ])

    # ------------------------------------------------------------ 2. 함수 36 개
    rows = [(s.replace("aero/coefficient/", ""), d["fn:" + s], fns[:, i])
            for i, s in enumerate(FN_NAMES)]
    res = errtable("== aero/coefficient/* 36 개 (단위 lbf, 모멘트축 lbf*ft) ==", rows)

    # ------------------------------------------------------------- 3. 축 합계
    axis_ref = {a: np.zeros(args.n) for a in AXES}
    for i, s in enumerate(FN_NAMES):
        axis_ref[FN_AXIS[i]] += d["fn:" + s]
    errtable("== 축 합계 ==", [
        ("DRAG", axis_ref["DRAG"], out["drag"].double().cpu().numpy()),
        ("SIDE", axis_ref["SIDE"], out["side"].double().cpu().numpy()),
        ("LIFT", axis_ref["LIFT"], out["lift"].double().cpu().numpy()),
        ("ROLL (about AERORP)", axis_ref["ROLL"], out["l"].double().cpu().numpy()),
        ("PITCH (about AERORP)", axis_ref["PITCH"], out["m"].double().cpu().numpy()),
        ("YAW (about AERORP)", axis_ref["YAW"], out["n"].double().cpu().numpy()),
    ])

    # --------------------------------------------------- 4. 동체축 힘 / 모멘트
    cg = np.stack([d["cgx"], d["cgy"], d["cgz"]], 1)
    np_out = {k: out[k].double().cpu().numpy() for k in ("fbx", "fby", "fbz", "l", "m", "n")}
    l_cg, m_cg, n_cg = aero.moments_about_cg(np_out, cg)
    fx, fy, fz = np_out["fbx"], np_out["fby"], np_out["fbz"]
    errtable("== 동체·풍축 힘과 CG 둘레 모멘트 (JSBSim forces/*, moments/*) ==", [
        ("forces/fwx-aero-lbs", d["fwx"], out["fwx"].double().cpu().numpy()),
        ("forces/fwy-aero-lbs", d["fwy"], out["fwy"].double().cpu().numpy()),
        ("forces/fwz-aero-lbs", d["fwz"], out["fwz"].double().cpu().numpy()),
        ("forces/fbx-aero-lbs", d["fbx"], fx),
        ("forces/fby-aero-lbs", d["fby"], fy),
        ("forces/fbz-aero-lbs", d["fbz"], fz),
        ("moments/l-aero-lbsft", d["Ml"], l_cg),
        ("moments/m-aero-lbsft", d["Mm"], m_cg),
        ("moments/n-aero-lbsft", d["Mn"], n_cg),
    ])

    # ------------------------------------------------------------------ 5. 대기
    Tr, P, rho, a = atm(T("h"))
    errtable("== 표준대기 ==", [
        ("atmosphere/T-R", d["T"], Tr.double().cpu().numpy()),
        ("atmosphere/P-psf", d["P"], P.double().cpu().numpy()),
        ("atmosphere/rho-slugs_ft3", d["rho"], rho.double().cpu().numpy()),
        ("atmosphere/a-fps", d["a"], a.double().cpu().numpy()),
        ("density-altitude = h ?", d["dalt"], d["h"]),
    ])

    # ------------------------------------------------------------------ 6. 추력
    n2n = (d["n2"] - eng.idle_n2) / eng.n2_factor
    aug = np.clip(d["thr_pos"] - 1.0, 0.0, 1.0)
    th = eng.thrust(torch.as_tensor(n2n, dtype=dtype, device=dev),
                    torch.as_tensor(aug, dtype=dtype, device=dev),
                    T("mach"), T("dalt")).double().cpu().numpy()
    errtable("== 추력 (JSBSim 의 N2 를 그대로 넣어 표만 대조) ==", [
        ("engine thrust-lbs", d["thrust"], th),
    ])

    spool_check(eng, dtype, dev)

    # 점별 상대오차는 값이 0 을 지나는 곳(예: Cnb_M 은 mach 1.0 에서 정확히 0)
    # 에서 의미가 없다.  판정은 **풀스케일 대비 최대 절대오차**로 한다.
    print("\n== 판정: 풀스케일 대비 최대오차 (abs max / |max|) ==")
    worst = sorted(((r[4] / max(r[1], 1e-30), r[0], r[2]) for r in res),
                   reverse=True)[:5]
    for fs, name, relmax in worst:
        print(f"  {name:12s} {fs:.3e}   (점별 상대 최대 {relmax:.2e})")
    lim = 1e-4 if dtype is torch.float32 else 1e-10
    bad = [w for w in worst if w[0] > lim]
    print("  -> " + ("전부 " if not bad else f"{[b[1] for b in bad]} 만 ")
          + f"{lim:g} 이내" + ("" if not bad else " 초과"))


def spool_check(eng, dtype, dev, dt=1.0 / 120.0):
    """스풀 동역학: 스로틀 계단 입력을 JSBSim 과 나란히 적분해 본다."""
    print("\n== 스풀 동역학 (스로틀 계단, 20,000 ft / M0.6) ==")
    for cmd0, cmd1 in ((0.5, 1.0), (1.0, 0.0), (0.0, 0.5)):
        fdm = new_fdm()
        fdm["ic/h-sl-ft"] = 20000.0
        fdm["ic/mach"] = 0.6
        for k in ("phi-deg", "theta-deg", "alpha-deg", "beta-deg", "gamma-deg",
                  "p-rad_sec", "q-rad_sec", "r-rad_sec"):
            fdm["ic/" + k] = 0.0
        fdm["fcs/throttle-cmd-norm"] = cmd0
        fdm.run_ic()
        fdm["propulsion/engine[0]/set-running"] = 1
        for _ in range(240):
            fdm.run()
        t = lambda x: torch.tensor([x], dtype=dtype, device=dev)
        N2 = t(fdm["propulsion/engine[0]/n2"])
        N2n = (N2 - eng.idle_n2) / eng.n2_factor
        fdm["fcs/throttle-cmd-norm"] = cmd1
        eN2, eT, jN2, jT = [], [], [], []
        for i in range(600):
            fdm.run()
            th, N2, N2n = eng.step(N2, N2n, t(cmd1), t(fdm["velocities/mach"]),
                                   t(fdm["atmosphere/density-altitude"]), dt)
            eN2.append(float(N2)); eT.append(float(th))
            jN2.append(fdm["propulsion/engine[0]/n2"])
            jT.append(fdm["propulsion/engine[0]/thrust-lbs"])
        eN2, eT = np.array(eN2), np.array(eT)
        jN2, jT = np.array(jN2), np.array(jT)
        print(f"  {cmd0:.1f} -> {cmd1:.1f} (5 s):  N2 최대오차 {np.abs(eN2-jN2).max():8.4f} %"
              f"   추력 최대오차 {np.abs(eT-jT).max():9.2f} lbf"
              f"  ({np.abs(eT-jT).max()/max(jT.max(),1):.2e} 상대)"
              f"   최종 {eT[-1]:.1f} vs {jT[-1]:.1f}")


# 함수 이름·축은 npz 에서 가져온다 (XML 순서 = JSBSim 합산 순서)
def _fn_meta():
    from jsbsim_f16_cuda.parse_f16 import unflatten
    with np.load(os.path.join(os.path.dirname(__file__), "f16_tables.npz")) as z:
        m = unflatten(z)
    names, axes = [], []
    for a in AXES:
        for fn in m["axes"][a]:
            names.append(str(fn["name"])); axes.append(a)
    return names, axes


FN_NAMES, FN_AXIS = _fn_meta()

if __name__ == "__main__":
    main()
