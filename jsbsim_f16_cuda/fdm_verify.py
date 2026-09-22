# SPDX-License-Identifier: GPL-3.0-or-later
"""비행모델을 JSBSim 과 **물리 프레임(1/120 s) 단위**로 대조한다.

여러 초를 돌려 궤적이 얼마나 벌어지는지 보는 지표는 혼돈에 잠겨서(같은 JSBSim 을
1e-6 kt 다르게 출발시켜도 수백 초 뒤에는 km 단위로 갈린다) **어느 항이 틀렸는지**를
못 짚는다.  여기서는 반대로 한다:

    JSBSim 의 한가운데 상태를 GPU 플랜트에 **통째로 심고 한 프레임만** 전진시켜
    중간량을 전부 대조한다.  누적이 없으니 남는 차이는 전부 모델 차이다.

두 엔진에 같은 원시 조종입력(aileron, elevator, rudder, throttle --
`f16_reference.JSBSimF16Ref.set_controls` 가 받는 그 값)을 물리 프레임마다 똑같이
먹이므로, 어긋나면 그것은 FCS·공력·추진·질량·적분기 중 하나다.

프레임 정렬 (이걸 틀리면 멀쩡한 모델이 틀려 보인다)
---------------------------------------------------
JSBSim `run()` 한 번의 결과를 읽으면 **행 k = (상태 S_k, 그 S_k 에서 계산한 힘)**
이다.  그 힘이 S_k -> S_{k+1} 을 만든다.  `F16Stick._substep` 도 같다: 들고 있는
상태에서 힘을 재고 그 힘으로 적분한다.  따라서 행 W 의 상태를 심으면

    GPU 프레임 j 의 힘·FCS·보조량  <->  JSB 행 W+j
    GPU 프레임 j 의 적분 결과 상태   <->  JSB 행 W+j+1

심어야 하는 것
--------------
상태만으로는 부족하다.  둘 다 한 프레임 이상의 이력을 들고 있다:

    rb.pos_ned/uvw/quat/pqr         행 W
    rb._hist["uvw"]/["pos"]         Adams-Bashforth 과거 미분값.  `_push` 가
                                    [2]<-[1], [1]<-[0] 순으로 미므로 **[0] 에
                                    직전 프레임**을 넣어야 한다
    flcs 지연버퍼 9 개 + 면 위치 3 개  행 W-1 (FLCS 가 스스로 한 프레임 미룬다)
    n2, ff_pps                      행 W-1 (`turbine.step` 이 W 로 올린다)
    a_body, wdot_i                  행 W-1 (조종석 하중배수가 한 프레임 지연)
    탱크 4 개                        행 W-1 (질량은 직전 프레임까지 태운 연료로 잰다)
    탱크 항 기준 무게중심            행 W-2 의 탱크로 잰 무게중심
    pitch_trim                      행 W

쓰는 법
-------
    python -m jsbsim_f16_cuda.fdm_verify                   # 전체 표 (13 입력열 x 4 조건)
    python -m jsbsim_f16_cuda.fdm_verify --check           # 회귀 검사 (임계 넘으면 exit 1)
    python -m jsbsim_f16_cuda.fdm_verify --check --lat 60  # 다른 위도 (중력·자전 항)
    python -m jsbsim_f16_cuda.fdm_verify --dtype float32   # float32 정밀도로
    python -m jsbsim_f16_cuda.fdm_verify --prog ail_step --trace wdot,cgz

이 방법으로 찾아 고친 것 (`docs/porting_notes.md`, 방법과 결과는 `docs/verification.md`):

    코리올리 계수 2 누락        v_dot 3.4e-2 -> 1.4e-6 ft/s^2
    자전 원심가속도 누락        w_dot 7.0e-2 -> 2.0e-4 ft/s^2
    질량·무게중심을 연료 합계 표로 보간  ->  탱크 점질량 모델 (관성 3e-11)
    연료유량 변화율 제한 누락    200 초 누적 +300 lb -> +9 lb
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from jsbsim_f16_cuda import f16_core as FC  # noqa: E402
from jsbsim_f16_cuda import rbdyn as RB  # noqa: E402
from jsbsim_f16_cuda.f16_reference import JSBSimF16Ref  # noqa: E402

FT = 0.3048
R_EARTH = 6371000.0
DT = 1.0 / 120.0
#: `--negative` (음성 대조).  `seed()` 가 읽는다.
NEGATIVE = False
LBF, LBFFT, SLUG, SLUGFT2 = FC.LBF, FC.LBFFT, FC.SLUG, FC.SLUGFT2

#: 읽어 오는 JSBSim 프로퍼티.  이름은 GPU 쪽 기록 키와 맞춘다.
JSB = dict(
    u="velocities/u-fps", v="velocities/v-fps", w="velocities/w-fps",
    p="velocities/p-rad_sec", q="velocities/q-rad_sec", r="velocities/r-rad_sec",
    phi="attitude/phi-rad", theta="attitude/theta-rad", psi="attitude/psi-rad",
    h="position/h-sl-ft", lat="position/lat-geod-deg", lon="position/long-gc-rad",
    vn="velocities/v-north-fps", ve="velocities/v-east-fps",
    vd="velocities/v-down-fps",
    alpha="aero/alpha-rad", beta="aero/beta-rad", mach="velocities/mach",
    qbar="aero/qbar-psf", vt="velocities/vt-fps", vc="velocities/vc-kts",
    vg="velocities/vg-fps",
    p_aero="velocities/p-aero-rad_sec", q_aero="velocities/q-aero-rad_sec",
    r_aero="velocities/r-aero-rad_sec",
    # 공력이 실제로 먹는 면 (`flcs_verify.TARGETS` 의 매핑)
    ail_pos="fcs/aileron-control", ele_pos="fcs/elevator-pos-rad",
    rud_pos="fcs/rudder-pos-rad", lef_pos="fcs/lef-pos-rad",
    sb_pos="fcs/speedbrake-pos-rad", flap_mix="fcs/flaperon-mix-rad",
    ail_position="fcs/aileron-position", tef="fcs/tef-control",
    e_roll="fcs/roll-trim-error", e_pitch="fcs/pitch-trim-error",
    e_yaw="fcs/yaw-trim-error", ptrim="fcs/pitch-trim-cmd-norm",
    n2="propulsion/engine[0]/n2", thrust="forces/fbx-prop-lbs",
    fuel="propulsion/total-fuel-lbs",
    ff="propulsion/engine[0]/fuel-flow-rate-pps",
    mass="inertia/mass-slugs", ixx="inertia/ixx-slugs_ft2",
    iyy="inertia/iyy-slugs_ft2", izz="inertia/izz-slugs_ft2",
    ixz="inertia/ixz-slugs_ft2", cgx="inertia/cg-x-in", cgz="inertia/cg-z-in",
    fbx="forces/fbx-total-lbs", fby="forces/fby-total-lbs",
    fbz="forces/fbz-total-lbs",
    l="moments/l-total-lbsft", m="moments/m-total-lbsft",
    n="moments/n-total-lbsft",
    t0="propulsion/tank[0]/contents-lbs", t1="propulsion/tank[1]/contents-lbs",
    t2="propulsion/tank[2]/contents-lbs", t3="propulsion/tank[3]/contents-lbs",
    fbx_a="forces/fbx-aero-lbs", fby_a="forces/fby-aero-lbs",
    fbz_a="forces/fbz-aero-lbs",
    l_a="moments/l-aero-lbsft", m_a="moments/m-aero-lbsft",
    n_a="moments/n-aero-lbsft",
    npx="accelerations/n-pilot-x-norm", npy="accelerations/n-pilot-y-norm",
    npz="accelerations/n-pilot-z-norm",
    pidot="accelerations/pidot-rad_sec2", qidot="accelerations/qidot-rad_sec2",
    ridot="accelerations/ridot-rad_sec2",
    udot="accelerations/udot-ft_sec2", vdot="accelerations/vdot-ft_sec2",
    wdot="accelerations/wdot-ft_sec2",
    pdot="accelerations/pdot-rad_sec2", qdot="accelerations/qdot-rad_sec2",
    rdot="accelerations/rdot-rad_sec2",
)


def _snap(fdm) -> dict:
    return {k: float(fdm[p]) for k, p in JSB.items()}


# ---------------------------------------------------------------- 입력 프로그램

def _rand(seed: int, hold: int = 8):
    rng = np.random.default_rng(seed)
    tab = rng.uniform(-1.0, 1.0, (400, 4))
    tab[:, 3] = rng.uniform(0.0, 1.0, 400)
    return lambda k: tuple(tab[min(k // hold, 399)])


PROGS = {
    "hold": lambda k: (0.0, 0.0, 0.0, 0.5),
    "ail_step": lambda k: (0.0 if k < 12 else 1.0, 0.0, 0.0, 0.5),
    "ele_step": lambda k: (0.0, 0.0 if k < 12 else 0.8, 0.0, 0.5),
    "ele_doublet": lambda k: (0.0, (0.0 if k < 12 else
                                    (0.5 if k < 36 else
                                     (-0.5 if k < 60 else 0.0))), 0.0, 0.5),
    "rud_step": lambda k: (0.0, 0.0, 0.0 if k < 12 else 0.6, 0.5),
    "thr_ab": lambda k: (0.0, 0.0, 0.0, 0.0 if k < 12 else 1.0),
    "thr_idle": lambda k: (0.0, 0.0, 0.0, 0.5 if k < 12 else 0.0),
    "combo": lambda k: (0.0 if k < 12 else 0.8, 0.0 if k < 12 else 0.6, 0.0, 0.9),
    "ramp": lambda k: (0.0, min(1.0, k / 120.0), 0.0, 0.5),
    "full": lambda k: (1.0 if k >= 12 else 0.0, 1.0 if k >= 12 else 0.0,
                       1.0 if k >= 12 else 0.0, 1.0),
    "rand0": _rand(0), "rand1": _rand(1), "rand2": _rand(2),
}

CONDS = ((450.0, 20000.0), (350.0, 10000.0), (600.0, 30000.0), (320.0, 5000.0))


# ---------------------------------------------------------------- JSBSim

def make_ref(kind: str, h0: float, lat: float) -> JSBSimF16Ref:
    if kind == "std":
        return JSBSimF16Ref(dt_physics=DT, h0_ft=h0, lat0_deg=lat)


def run_jsb(prog, frames, v0, h0, kind="std", lat=0.0):
    be = make_ref(kind, h0, lat)
    be.reset(x=0.0, y=0.0, psi=0.0, v_kt=v0, h_ft=h0)
    fdm = be.fdm
    rows = []
    for k in range(frames):
        be.set_controls(*prog(k))
        be.run_one()
        rows.append(_snap(fdm))
    return rows, be


# ---------------------------------------------------------------- GPU



def seed(dyn, rows, W, prog, lat0, lon0):
    """JSB 행 W 의 상태(와 한 프레임 이력)를 플랜트에 심는다.  `lat0/lon0` [rad] 는 평면 원점."""
    dev, dt = dyn.device, dyn.dtype
    t = lambda x: torch.full((dyn.N,), float(x), device=dev, dtype=dt)
    cur, prv, pr2, pr3 = rows[W], rows[W - 1], rows[W - 2], rows[W - 3]
    dyn.rb.pos_ned.copy_(torch.stack((
        t((math.radians(cur["lat"]) - lat0) * R_EARTH),
        t((cur["lon"] - lon0) * R_EARTH * math.cos(lat0)),
        t(-cur["h"] * FT)), -1))
    dyn.rb.uvw.copy_(torch.stack((t(cur["u"] * FT), t(cur["v"] * FT),
                                  t(cur["w"] * FT)), -1))
    dyn.rb.pqr.copy_(torch.stack((t(cur["p"]), t(cur["q"]), t(cur["r"])), -1))
    dyn.rb.quat.copy_(RB.quat_from_euler(t(cur["phi"]), t(cur["theta"]),
                                         t(cur["psi"])))
    # Adams-Bashforth 이력.  `_push` 가 한 칸 밀므로 **[0] 이 직전 프레임**이다.
    hu, hp = dyn.rb._hist["uvw"], dyn.rb._hist["pos"]
    for slot, row in ((0, prv), (1, pr2), (2, pr3)):
        hu[slot].copy_(torch.stack((t(row["udot"] * FT), t(row["vdot"] * FT),
                                    t(row["wdot"] * FT)), -1))
        hp[slot].copy_(torch.stack((t(row["vn"] * FT), t(row["ve"] * FT),
                                    t(row["vd"] * FT)), -1))
    dyn.rb._fresh.fill_(False)
    dyn.n2.copy_(t(prv["n2"]))
    dyn.n2norm.copy_((dyn.n2 - dyn.turbine.idle_n2) / dyn.turbine.n2_factor)
    dyn.pitch_trim.copy_(t(cur["ptrim"]))
    if getattr(dyn, "_tank", False):
        # 탱크 모델: 질량은 직전 프레임까지 태운 연료(행 W-1), 탱크 평행축 항은 그
        # 전 프레임 무게중심(행 W-2 탱크) 기준이다 -- JSBSim 모델 실행 순서 그대로.
        tank = lambda row: torch.tensor([[row[f"t{i}"] for i in range(4)]] * dyn.N,
                                        device=dev, dtype=dt)
        dyn.fuel.copy_(tank(prv))
        dyn._cg_tank.copy_(dyn.mass.props(tank(pr2))[1])
    else:
        dyn.fuel.copy_(t(cur["fuel"]))
    dyn.flcs.reset(
        alpha_rad=t(prv["alpha"]), p_aero=t(prv["p_aero"]),
        q_aero=t(prv["q_aero"]), r_aero=t(prv["r_aero"]),
        n_pilot_z_norm=t(prv["npz"]), n_pilot_y_norm=t(prv["npy"]),
        mach=t(prv["mach"]), vc_kts=t(prv["vc"]), vg_fps=t(prv["vg"]),
        ail_pos_norm=t(prv["ail_position"]),
        ele_pos_norm=t(prv["ele_pos"] / 0.436),
        tef_control=t(prv["tef"]))
    dyn.flcs.e_roll_prev.copy_(t(prv["e_roll"]))
    dyn.flcs.e_pitch_prev.copy_(t(prv["e_pitch"]))
    dyn.flcs.e_yaw_prev.copy_(t(prv["e_yaw"]))
    dyn.a_body.copy_(torch.stack((t(prv["fbx"] / prv["mass"]),
                                  t(prv["fby"] / prv["mass"]),
                                  t(prv["fbz"] / prv["mass"])), -1))
    dyn.wdot_i.copy_(torch.stack((t(prv["pidot"]), t(prv["qidot"]),
                                  t(prv["ridot"])), -1))
    if hasattr(dyn, "ff_pps"):
        dyn.ff_pps.copy_(t(prv["ff"]))
    dyn._engine_trim.zero_()          # 트림 프레임이 아니다
    if hasattr(dyn, "_trim_frame"):
        dyn._trim_frame.zero_()
    if NEGATIVE:
        # 음성 대조: 실제로 있었던 버그 하나(자전 원심가속도 연직 몫 누락, g 의 0.2 %)를
        # 일부러 되살린다.  대조가 이것을 못 잡으면 대조가 쓸모없는 것이다.
        dyn.rb.a_cent_ned.zero_()


#: `_substep` 의 carry.  `ff_pps` 는 2026-09-23 에 생겼다 -- 그 전 커밋에도
#: 이 파일을 그대로 씌워 보려고 있으면 넣는다 (그게 회귀 검사의 요점이다).
def _carry_names(dyn):
    if isinstance(dyn, FC.F16Stick):
        return ("n_pilot", "n2", "n2norm", "fuel", "a_body", "wdot_i", "ff_pps")


def make_plant(plant: str, ref: str, be: JSBSimF16Ref, rows, W, dtype, device):
    if plant == "stick" and ref == "std":
        return FC.F16Stick(1, device=device, dtype=dtype, lat0_deg=be.lat0_deg)
    raise ValueError((plant, ref))


def run_gpu(prog, frames, rows, W, dtype, device="cpu", plant="stick", ref="std", be=None):
    dyn = make_plant(plant, ref, be, rows, W, dtype, device)
    seed(dyn, rows, W, prog, be._lat0, be._lon0)
    stick_of = lambda k: torch.tensor([prog(k)] * dyn.N, device=dyn.device, dtype=dtype)

    out, cur = [], {}
    f_flcs, f_turb, f_aero = dyn.flcs.step, dyn.turbine.step, dyn.aero
    f_mcg, f_der, f_rb = dyn._moments_about_cg, dyn.rb.derivatives, dyn.rb.step

    def flcs(*a, **kw):
        cur.update(alpha=float(kw["alpha_rad"][0]), mach=float(kw["mach"][0]),
                   vc=float(kw["vc_kts"][0]), vg=float(kw["vg_fps"][0]),
                   npz=float(kw["n_pilot_z_norm"][0]),
                   npy=float(kw["n_pilot_y_norm"][0]))
        o = f_flcs(*a, **kw)
        cur.update(ele_pos=float(o.elevator_pos_rad[0]),
                   ail_pos=float(o.aileron_pos_rad[0]),
                   rud_pos=float(o.rudder_pos_rad[0]),
                   lef_pos=float(o.lef_pos_rad[0]),
                   sb_pos=float(o.speedbrake_pos_rad[0]),
                   flap_mix=float(o.flaperon_mix_rad[0]))
        return o

    def turb(*a, **kw):
        th, n2, n2n = f_turb(*a, **kw)
        cur.update(thrust=float(th[0]), n2=float(n2[0]))
        return th, n2, n2n

    def aero(st):
        cur.update(qbar=float(st.qbar[0]), beta=float(st.beta[0]))
        o = f_aero(st)
        cur.update(fbx_a=float(o["fbx"][0]), fby_a=float(o["fby"][0]),
                   fbz_a=float(o["fbz"][0]))
        return o

    def mcg(o, cg_in):
        l, m, n = f_mcg(o, cg_in)
        cur.update(l_a=float(l[0]), m_a=float(m[0]), n_a=float(n[0]),
                   cgx=float(cg_in[0, 0]), cgz=float(cg_in[0, 2]))
        return l, m, n

    def der(*a, **kw):
        o = f_der(*a, **kw)
        _, uvw_dot, _, pqr_dot = o
        cur.update(udot=float(uvw_dot[0, 0]) / FT,
                   vdot=float(uvw_dot[0, 1]) / FT,
                   wdot=float(uvw_dot[0, 2]) / FT,
                   pdot=float(pqr_dot[0, 0]), qdot=float(pqr_dot[0, 1]),
                   rdot=float(pqr_dot[0, 2]))
        return o

    def rb(wrench, mp):
        cur.update(fbx=float(wrench.force[0, 0]) / LBF,
                   fby=float(wrench.force[0, 1]) / LBF,
                   fbz=float(wrench.force[0, 2]) / LBF,
                   l=float(wrench.moment[0, 0]) / LBFFT,
                   m=float(wrench.moment[0, 1]) / LBFFT,
                   n=float(wrench.moment[0, 2]) / LBFFT,
                   mass=float(mp.mass[0]) / SLUG,
                   ixx=float(mp.ixx[0]) / SLUGFT2,
                   iyy=float(mp.iyy[0]) / SLUGFT2,
                   izz=float(mp.izz[0]) / SLUGFT2,
                   ixz=float(mp.ixz[0]) / SLUGFT2,
                   fuel=float(dyn.fuel[0].sum()))
        res = f_rb(wrench, mp)
        phi, theta, psi = RB.euler_from_quat(dyn.rb.quat)
        vned = dyn.rb.velocity_ned()
        cur.update(u=float(dyn.rb.uvw[0, 0]) / FT,
                   v=float(dyn.rb.uvw[0, 1]) / FT,
                   w=float(dyn.rb.uvw[0, 2]) / FT,
                   p=float(dyn.rb.pqr[0, 0]), q=float(dyn.rb.pqr[0, 1]),
                   r=float(dyn.rb.pqr[0, 2]), phi=float(phi[0]),
                   theta=float(theta[0]), psi=float(psi[0]),
                   h=-float(dyn.rb.pos_ned[0, 2]) / FT,
                   vn=float(vned[0, 0]) / FT, ve=float(vned[0, 1]) / FT,
                   vd=float(vned[0, 2]) / FT)
        out.append(dict(cur))
        cur.clear()
        return res

    (dyn.flcs.step, dyn.turbine.step, dyn.aero, dyn._moments_about_cg,
     dyn.rb.derivatives, dyn.rb.step) = flcs, turb, aero, mcg, der, rb

    names = _carry_names(dyn)
    carry = tuple(getattr(dyn, n) for n in names)
    for j in range(frames):
        if isinstance(dyn, FC.F16Stick):
            carry = dyn._substep(*carry, stick=stick_of(W + j))
        for n, val in zip(names, carry):
            getattr(dyn, n).copy_(val)
        carry = tuple(getattr(dyn, n) for n in names)
    return out


def run_fused(prog, frames, rows, W, dtype, device="cuda", plant="stick", ref="std", be=None):
    """같은 대조를 **CUDA 융합 커널**로 (`fused_core.py`).

    심는 것은 `seed()` 그대로다 -- 커널이 torch 플랜트의 상태 텐서를 제자리에서 읽고
    쓰기 때문이다.  중간량은 커널의 디버그 버퍼에서 이 파일의 키 이름 그대로 읽는다.

    F16Stick: 프레임 j 의 힘은 스틱 prog(W+j) 로 잰다 (`run_gpu` 가 `_substep` 에 주는
    것과 같다).  커널은 **들고 있던** 스틱으로 힘을 재므로 프레임마다 그 버퍼에 넣고 한
    프레임씩 부른다.
    """
    from jsbsim_f16_cuda import fused_core as FK
    dyn = make_plant(plant, ref, be, rows, W, dtype, "cuda")
    seed(dyn, rows, W, prog, be._lat0, be._lon0)
    out = []
    if isinstance(dyn, FC.F16Stick):
        fz = FK.FusedStick(dyn)
        for j in range(frames):
            u = torch.tensor([prog(W + j)] * dyn.N, dtype=dtype, device="cuda")
            dyn.stick.copy_(u)
            dbg = torch.zeros(dyn.N, 1, FK.NDBG, dtype=dtype, device="cuda")
            fz.step(u, 1, dbg=dbg)
            d = dbg[0, 0].double().cpu().numpy()
            out.append({k: float(d[i]) for i, k in enumerate(FK.DBG_KEYS)})
        return out


# ---------------------------------------------------------------- 대조

#: 힘·FCS·보조량·질량 -- GPU 프레임 j 가 JSB 행 W+j 와 짝이다.
FORCE = ("ele_pos", "ail_pos", "rud_pos", "lef_pos", "sb_pos", "flap_mix",
         "thrust", "n2", "fuel", "mass", "ixx", "iyy", "izz", "ixz",
         "cgx", "cgz", "alpha", "beta", "mach", "qbar", "vc", "vg",
         "npz", "npy", "fbx_a", "fby_a", "fbz_a", "l_a", "m_a", "n_a",
         "fbx", "fby", "fbz", "l", "m", "n",
         "udot", "vdot", "wdot", "pdot", "qdot", "rdot")
#: 적분 결과 -- GPU 프레임 j 가 JSB 행 W+j+1 과 짝이다.
STATE = ("u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "h",
         "vn", "ve", "vd")

#: `--check` 의 임계.  한 프레임 |오차| (단위는 각 양의 단위).
#: 2026-09-23 실측 최대값의 대략 3 배로 잡았다 -- 회귀만 잡고 잡음은 안 잡는다.
LIMITS = {
    "udot": 1e-3, "vdot": 2e-4, "wdot": 1e-3,          # ft/s^2
    "pdot": 2e-4, "qdot": 1e-3, "rdot": 2e-4,          # rad/s^2
    "ele_pos": 1e-9, "ail_pos": 1e-9, "rud_pos": 1e-9,  # rad, 정확히 0 이어야
    "thrust": 1e-3, "n2": 1e-6,
}


def compare(jrows, grows, W, n):
    out = {}
    for k in FORCE + STATE:
        off = 1 if k in FORCE else 0
        a = np.array([grows[i][k] for i in range(n)])
        b = np.array([jrows[W + 1 + i - off][k] for i in range(n)])
        sc = max(np.abs(b).max(), 1e-12)
        out[k] = (np.abs(a - b), np.abs(a - b) / sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1,
                    help="심은 뒤 몇 프레임을 대조할까 (기본 1 = 순수 한 스텝)")
    ap.add_argument("--warm", type=int, default=36, help="어느 프레임에서 심을까")
    ap.add_argument("--prog", default="")
    ap.add_argument("--cond", default="")
    ap.add_argument("--dtype", default="float64",
                    choices=("float64", "float32"))
    ap.add_argument("--trace", default="")
    ap.add_argument("--check", action="store_true", help="임계 넘으면 exit 1")
    ap.add_argument("--lat", type=float, default=0.0, help="std 기준의 위도 [deg]")
    ap.add_argument("--negative", action="store_true",
                    help="음성 대조: 원심가속도 항을 일부러 빼고 대조가 그것을 잡는지 본다")
    ap.add_argument("--fused", action="store_true",
                    help="torch 플랜트 대신 CUDA 융합 커널을 대조한다 (cuda 필요)")
    a = ap.parse_args()
    global NEGATIVE
    NEGATIVE = a.negative
    plant = getattr(a, "plant", "stick")
    ref = getattr(a, "ref", "") or ("std" if plant == "stick" else "gym")

    dtype = dict(float64=torch.float64, float32=torch.float32)[a.dtype]
    conds = CONDS
    if a.cond:
        v, h = a.cond.split(",")
        conds = [(float(v), float(h))]
    progs = [a.prog] if a.prog else list(PROGS)

    worst = {k: 0.0 for k in LIMITS}
    where = {k: "" for k in LIMITS}
    print("=" * 78)
    print(f"비행모델 프레임 단위 대조 (원시 조종입력, dtype={a.dtype}, "
          f"심은 프레임 {a.warm}, {a.frames} 프레임 대조, 플랜트 {plant}, 기준 {ref}"
          + (f", 위도 {a.lat:g}" if ref == "std" else "")
          + (", CUDA 융합 커널" if a.fused else "") + ")")
    print("=" * 78)
    for v0, h0 in conds:
        print(f"\n--- v0={v0:.0f} kt  h0={h0:.0f} ft ---")
        print(f"  {'입력열':<14}{'|du.|':>10}{'|dv.|':>10}{'|dw.|':>10}"
              f"{'|dp.|':>10}{'|dq.|':>10}{'|dr.|':>10}{'FCS':>8}")
        for pn in progs:
            prog = PROGS[pn]
            jr, be = run_jsb(prog, a.warm + a.frames + 2, v0, h0, ref, a.lat)
            run = run_fused if a.fused else run_gpu
            gr = run(prog, a.frames, jr, a.warm, dtype, plant=plant, ref=ref, be=be)
            e = compare(jr, gr, a.warm, a.frames)
            fcs = max(e[k][0].max() for k in
                      ("ele_pos", "ail_pos", "rud_pos", "lef_pos", "sb_pos"))
            print(f"  {pn:<14}" + "".join(
                f"{e[k][0].max():>10.2e}" for k in
                ("udot", "vdot", "wdot", "pdot", "qdot", "rdot"))
                + f"{fcs:>8.1e}")
            for k in LIMITS:
                if e[k][0].max() > worst[k]:
                    worst[k] = e[k][0].max()
                    where[k] = f"{pn} @ {v0:.0f}kt/{h0:.0f}ft"
            if a.trace:
                for k in a.trace.split(","):
                    off = 1 if k in FORCE else 0
                    print(f"    --- {k} ---")
                    for i in range(a.frames):
                        g, j = gr[i][k], jr[a.warm + 1 + i - off][k]
                        print(f"      f{i:<3} GPU {g:+18.10f}  JSB {j:+18.10f}"
                              f"  d {g - j:+.3e}")

    print("\n" + "=" * 78)
    print(f"  {'양':<10}{'최대 |오차|':>14}{'임계':>12}   어디서")
    bad = 0
    for k, lim in LIMITS.items():
        flag = "" if worst[k] <= lim else "   <-- 임계 초과"
        bad += worst[k] > lim
        print(f"  {k:<10}{worst[k]:>14.3e}{lim:>12.0e}   {where[k]}{flag}")
    if a.negative:
        # 일부러 넣은 버그를 잡아야(= 임계를 넘어야) 성공이다.
        print(f"\n음성 대조: " + (f"잡았다 -- {bad} 개 항목이 임계를 넘었다 (정상)" if bad
                                  else "못 잡았다 (대조가 무디다)"))
        return 0 if bad else 1
    if a.check:
        print(f"\n{'통과' if not bad else f'실패 -- {bad} 개 항목이 임계를 넘었다'}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
