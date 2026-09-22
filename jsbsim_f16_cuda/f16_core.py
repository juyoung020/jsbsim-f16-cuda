# SPDX-License-Identifier: GPL-3.0-or-later
"""F-16 플랜트 -- 조종간 입력만으로 도는 배치 시뮬레이터.

JSBSim 1.3 의 F-16 모델을 한 프레임(1/120 s) 단위로 그대로 옮긴 계층들을 묶는다.

    FGAuxiliary 보조량   받음각·옆미끄럼각·마하·동압·교정대기속도, 조종석 하중배수
    연료유량             FGTurbine 연료소모와 그 변화율 제한 (`Seek`)
    TankMass             f16.xml 탱크 네 개(점질량)로 질량·관성·무게중심, 소모·급유
    TrimGrid             JSBSim `do_trim(1)` 을 (고도 x 속도 x 연료) 격자로 잰 표
    F16Stick             위 전부 + FLCS(`flcs.py`) + 공력(`aero.py`) + 엔진(`propulsion.py`)
                         + 6-DOF 적분(`rbdyn.py`) 을 한 프레임으로 묶은 플랜트

모든 상태는 배치 텐서이고 제자리로만 갱신한다 -- `F16Stick.step` 을 통째로 CUDA
그래프에 캡처할 수 있다 (`examples/bench.py`).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from jsbsim_f16_cuda import rbdyn as RB                                  # noqa: E402
from jsbsim_f16_cuda.aero import AeroState, F16Aero                      # noqa: E402
from jsbsim_f16_cuda.flcs import (BatchFLCS, ELEVATOR_RANGE_RAD,         # noqa: E402
                              MACH_HI, TEF_HI_MACH_RAD, TEF_NORM_GAIN,
                              TEF_VC_KT)
from jsbsim_f16_cuda.propulsion import F100Turbine, StdAtmosphere        # noqa: E402

FT = 0.3048
KT_TO_MS = 0.514444
FPS_PER_KT = 1.6878098571
LBF = RB.LBF
LBFFT = RB.LBFFT
SLUG = RB.SLUG
SLUGFT2 = RB.SLUGFT2

PHYSICS_HZ = 120.0
DT_PHYS = 1.0 / PHYSICS_HZ

# --- 표준대기 해면값 (propulsion.py 와 같은 상수에서 유도) --------------------
P_SL_PSF = 2116.228
R_ENG = 1716.55716
RHO_SL = P_SL_PSF / (R_ENG * 518.67)
A_SL_FPS = math.sqrt(1.4 * P_SL_PSF / RHO_SL)         # 1116.4486 ft/s
#: 하중배수를 나눌 때 쓰는 **해면 표준중력** [ft/s^2].  국소 중력을 쓰면
#: 20,000 ft 에서 잔차가 두 자릿수 나빠진다.
G_STD_FPS2 = 32.174049

#: 조종석 눈점, 구조축 [in] (f16.xml `EYEPOINT`).  무게중심 앞 약 12 ft, 위 약 2.8 ft.
EYEPOINT_IN = (-336.2, 0.0, 29.5)



def fuel_flow_pps(thrust_lbf: torch.Tensor, aug_cmd: torch.Tensor,
                  n2norm: torch.Tensor) -> torch.Tensor:
    """`FGTurbine` 연료소모 [lb/s] 의 목표값.

        sfc = 2.0500                                  (AB -- 비례가 아니라 계단)
            = 0.6660 * (1 + 1.1907 * (1 - N2norm)^2)  (밀리터리)
        pph = max(thrust * sfc, 756.0)                (756 pph = 0.21 pps 아이들)

    JSBSim 21 점 **정상상태** 스윕에 소수 4 자리까지 맞춘 식이다.  AB 는 비례가
    아니라 계단이다.  스로틀을 흔든 과도 데이터로는 이 식을 확인할 수 없다 --
    추력·N2·연료유량의 위상이 서로 어긋나기 때문이다.

    **이것은 목표값이다.**  JSBSim 은 여기에 곧장 가지 않고 `fuel_flow_seek` 의
    변화율 제한을 거친다 -- 정상상태 스윕으로 맞춘 식이라 그 사실이 안 보였다.
    """
    sfc = torch.where(aug_cmd > 0.0,
                      torch.full_like(n2norm, 2.0500),
                      0.6660 * (1.0 + 1.1907 * (1.0 - n2norm) ** 2))
    return (thrust_lbf.clamp_min(0.0) * sfc).clamp_min(756.0) / 3600.0


#: 연료유량의 변화율 제한 [pph/s].  `FGTurbine` 의 `Seek`.
FF_RATE_UP_PPH_S = 5000.0
FF_RATE_DOWN_PPH_S = 10000.0


def fuel_flow_seek(ff_pps: torch.Tensor, target_pps: torch.Tensor,
                   dt: float = DT_PHYS) -> torch.Tensor:
    """연료유량은 목표로 **한 번에 못 간다** -- JSBSim `FGTurbine::Seek`.

    추력은 스로틀을 따라 한 프레임에 뛰는데(아이들 -706 -> AB 18,761 lbf, 실측)
    연료유량은 5,000 pph/s 로 기어 올라간다.  그래서 스풀 중에는 `thrust * sfc`
    가 실제 소모보다 한참 크다: 스로틀 1.0, N2 75 에서 JSBSim 0.26 pps 대
    목표식 10.6 pps (**40 배**).

    실측 기울기 (아이들<->AB 계단, 세 비행조건에서 같다):

        상승  +5,000.0 pph/s      하강  -10,000.0 pph/s

    안 넣으면 GPU 가 연료를 **더 먹는다** -- 연료가 곧 중량이라 기체 무게가
    어긋난다.  200 초 누적 실측 (같은 스로틀 열을 두 엔진에 먹여 연료유량을 적분):

        스로틀 열             JSBSim    목표식만    차
        무작위 1 Hz            451.5 lb   726.7 lb  +275.2 lb (+1.33 % 중량)
        사인 8 s (AB 경계 왕복) 448.7      767.4     +318.7    (+1.54 %)
        사인 30 s             1055.7     1261.4     +205.7    (+1.00 %)
        고정 0.55 (정상상태)    1601.9     1608.2      +6.3    (+0.03 %)

    마지막 줄이 요점이다 -- **정상상태에서는 목표식이 맞다.**  틀린 것은 과도구간
    뿐이고, 그래서 정상상태 스윕으로만 검증하면 안 보인다.
    """
    up = FF_RATE_UP_PPH_S * dt / 3600.0
    dn = FF_RATE_DOWN_PPH_S * dt / 3600.0
    return torch.clamp(target_pps, ff_pps - dn, ff_pps + up)




# =============================================================================
# 보조량 -- JSBSim FGAuxiliary
# =============================================================================

def pitot_total_pressure(mach: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """`FGAuxiliary::PitotTotalPressure`.  M<1 등엔트로픽, M>=1 라일리."""
    ms = mach.clamp_min(1e-6)
    sub = p * (1.0 + 0.2 * mach * mach) ** 3.5
    sup = (p * 166.92158009316827 * ms ** 7
           / (7.0 * ms * ms - 1.0).clamp_min(1e-9) ** 2.5)
    return torch.where(mach < 1.0, sub, sup)


def mach_from_impact_pressure(qc: torch.Tensor, p_sl: float,
                              iters: int = 12) -> torch.Tensor:
    """`FGAuxiliary::MachFromImpactPressure`.  초음속 가지는 고정점 반복.

    JSBSim 도 같은 반복을 돌린다.  수렴 판정 대신 회수를 고정했다 -- 판정은
    GPU->CPU 동기화라 CUDA 그래프가 깨진다.  12 회면 1e-12 로 수렴한다.
    """
    A = (qc / p_sl + 1.0).clamp_min(1e-9)
    sub = (5.0 * (A ** (2.0 / 7.0) - 1.0)).clamp_min(0.0).sqrt()
    m = torch.full_like(A, 2.0)
    for _ in range(iters):
        m = 0.881285 * (A * (1.0 - 1.0 / (7.0 * m * m)).clamp_min(1e-9) ** 2.5).sqrt()
    return torch.where(A < 1.89293, sub, m)


def vcalibrated_kts(mach: torch.Tensor, p_psf: torch.Tensor) -> torch.Tensor:
    """교정대기속도 [kt].  **충격압은 국소 정압, 역산은 해면 정압**이다."""
    pt = pitot_total_pressure(mach, p_psf)
    return mach_from_impact_pressure(pt - p_psf, P_SL_PSF) * (A_SL_FPS / FPS_PER_KT)


def tef_control_seed(vc_kts: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
    """리셋 시점의 `fcs/tef-control`.  `flcs.py:421` 의 3 분기를 그대로 푼다.

    **표에서 보간하면 안 된다 -- 스위치다.**  값이 세 개(0.9998 / 0 / -0.1)뿐인데
    속도로 선형보간하면 문턱을 넘는 칸에서 **어느 분기에도 없는 중간값**이 나온다
    (예: 325 kt / 20,000 ft 에서 0.4966, JSBSim 은 1.000).  뒷전 플랩 킨매틱이
    1/3 norm/s 라 그러면 **1.5 초 동안** 플랩이 절반만 펴진 채로 난다.
    """
    return torch.where(
        vc_kts < TEF_VC_KT, torch.full_like(vc_kts, 0.349 * TEF_NORM_GAIN),
        torch.where(mach > MACH_HI,
                    torch.full_like(vc_kts, TEF_HI_MACH_RAD * TEF_NORM_GAIN),
                    torch.zeros_like(vc_kts))).clamp(-1.0, 1.0)


def eye_arm_ft(cg_in: torch.Tensor) -> torch.Tensor:
    """무게중심 기준 조종석 눈점 `(N,3)` [ft, body].

    `FGMassBalance::StructuralToBody` -- 구조축은 x 가 뒤, z 가 위라 x 와 z 만
    부호가 뒤집힌다.  통째로 뒤집으면 크기는 맞고 부호만 틀려서 안 보인다.
    """
    ex, ey, ez = EYEPOINT_IN
    return torch.stack(((cg_in[..., 0] - ex) / 12.0,
                        (ey - cg_in[..., 1]) / 12.0,
                        (cg_in[..., 2] - ez) / 12.0), dim=-1)


def pilot_accel_g(a_body_prev: torch.Tensor, wdot_prev: torch.Tensor,
                  omega_now: torch.Tensor, r_eye: torch.Tensor) -> torch.Tensor:
    """조종석 하중배수 `(N,3)` [g].  `FGAccelerations::CalculatePilotAccel`.

        n = ( vBodyAccel(k-1) + wdot(k-1) x r + w(k) x (w(k) x r) ) / g_SL

    **입력마다 나이가 다르다.**  JSBSim 프레임 순서가
    `FCS -> Auxiliary -> 공력 -> ... -> Accelerations` 라, Auxiliary 안에서
    `vBodyAccel` 과 `vPQRidot` 은 직전 프레임 Accelerations 가 만든 값이고
    `vPQR` 은 이번 프레임 Propagate 가 방금 적분한 값이다.  실측 잔차:

        이 조합                    max 8.2e-7   p99 2.9e-7
        wdot 을 이번 프레임으로     max 4.0e+0   p99 1.3e-1
        각속도까지 직전 프레임으로   max 1.3e-1   p99 1.4e-2
        전부 이번 프레임            max 3.6e+0   p99 1.6e-1

    중앙값은 어느 조합이나 작아서 **고하중에서만 튄다** -- 그래서 틀려도 한참
    모른다.  `a_body` 는 **중력을 뺀** 공력+추력이라 수평비행에서 z 가 -1 이다
    (JSBSim `accelerations/n-pilot-z-norm` 과 같은 부호).

    레버암 두 항 모두 필요하다 -- 조종석이 CG 앞 12.03 ft·위 2.76 ft 라
    `wdot x r` 이 p99 0.52 g, 원심항이 p99 0.25 g 다.
    """
    ang = (torch.cross(wdot_prev, r_eye, dim=-1)
           + torch.cross(omega_now, torch.cross(omega_now, r_eye, dim=-1), dim=-1))
    return (a_body_prev + ang) / G_STD_FPS2


# =============================================================================
# 중력 -- JSBSim 1.3 기본 모델(WGS84 + J2)의 크기
# =============================================================================
#: JSBSim `FGInertial` 상수 (ft 단위).
_GM_FT3_S2 = 14.0764417572e15
_A_FT = 20925646.32546
_B_FT = 20855486.5951
_J2 = 1.0826266835531513e-3


def gravity_j2_ms2(lat_deg: float, h_ft: float) -> float:
    """측지 위도 `lat_deg`, 해면고도 `h_ft` 에서 JSBSim 중력(원심력 제외)의 크기 [m/s^2].

    `FGInertial::GetGravityJ2` 와 같은 식이다.  JSBSim `accelerations/gravity-ft_sec2`
    와 1e-8 (상대) 안에서 맞는다 (위도 0/37.6/60, 고도 5k~40k ft 실측).
    `RigidBody6DOF` 는 이 값을 `H_REF`(20,000 ft) 에서 한 번 받아 역제곱으로 늘인다.
    """
    e2 = 1.0 - (_B_FT / _A_FT) ** 2
    s, c = math.sin(math.radians(lat_deg)), math.cos(math.radians(lat_deg))
    n_rad = _A_FT / math.sqrt(1.0 - e2 * s * s)
    x, z = (n_rad + h_ft) * c, (n_rad * (1.0 - e2) + h_ft) * s
    r = math.hypot(x, z)
    sin_gc = z / r
    pre = 1.5 * _J2 * (_A_FT / r) ** 2
    g = _GM_FT3_S2 / (r * r)
    gx = -g * (1.0 + pre * (1.0 - 5.0 * sin_gc * sin_gc)) * x / r
    gz = -g * (1.0 + pre * (3.0 - 5.0 * sin_gc * sin_gc)) * z / r
    return math.hypot(gx, gz) * FT


# =============================================================================
# 탱크 질량 -- JSBSim 과 같은 규칙, 표 없음
# =============================================================================

class TankMass:
    """f16.xml 의 빈 기체 + 조종사 + 연료탱크 네 개(점질량)로 질량·관성·무게중심.

    `props(fuel)` / `advance(fuel, flow_pps)` 이고 `fuel` 은 `(N, 4)` 탱크별 [lb] 다.

    JSBSim 1.3 과 같은 식이다:

        무게중심  (W_e r_e + sum W_i r_i) / W          구조축 [in]
        관성      J_xml + sum m_i (평행축, 현재 CG 기준)  `FGMassBalance::GetPointmassInertia`
        소모      연료가 있는 급유 탱크에 **똑같이 나눠** 뺀다 (`FGPropulsion::ConsumeFuel`)
        급유      꽉 차지 않은(< 99.99 %) 탱크에 100 lb/s 를 똑같이 나눠 넣는다
                  (`FGPropulsion::DoRefuel`).  기본은 꺼져 있다.

    **순서도 JSBSim 과 같다**: 한 프레임의 질량은 직전 프레임까지 태운 연료로
    잰다 (`FGMassBalance` 가 `FGPropulsion` 보다 먼저 돈다).  `F16Stick` 이 프레임
    머리에서 `props`, 꼬리에서 `advance` 를 부르는 것이 그 순서다.  하나 더:
    탱크의 평행축 항은 `FGPropulsion` 이 **그때의 무게중심**(한 프레임 전 것)으로
    미리 계산해 두므로, 탱크 항만 직전 프레임 무게중심을 기준으로 한다 (`cg_tank`).
    이걸 빼면 관성이 1e-6 (상대) 어긋나고, 넣으면 3e-11 이다 (`f16_check.py mass`).

    연료가 전부 떨어지면 JSBSim 은 엔진을 끈다(추력 0, N2 감속).  **그것은 흉내 내지
    않는다** -- 탱크가 0 에 머물 뿐 엔진은 계속 돈다.
    """
    EMPTY_LBS = 17400.0
    CG_EMPTY_IN = (-193.0, 0.0, -5.1)
    #: (ixx, iyy, izz, ixz) [slug ft^2].  ixz 는 `inertia/ixz-slugs_ft2` 부호 그대로.
    J_EMPTY = (9496.0, 55814.0, 63100.0, -982.0)
    #: (무게 [lb], 위치 [in]).  f16.xml 의 `<pointmass name="Pilot">`.
    POINTMASSES = ((230.0, (-336.2, 0.0, 0.0)),)
    TANK_IN = ((-174.4, 65.0, 5.0), (-174.4, -65.0, 5.0),
               (-174.4, 65.0, -15.0), (-174.4, -65.0, -15.0))
    TANK_CAP_LBS = (3486.0, 3486.0, 2991.0, 2991.0)
    #: f16.xml 기본 탱크량 (내부 1,500 x 2).
    DEFAULT_LBS = (1500.0, 1500.0, 0.0, 0.0)
    #: `FGPropulsion` 의 RefuelRate 6,000 lb/min.
    REFUEL_PPS = 100.0

    def __init__(self, n: int, device, dtype, refuel: bool = False,
                 fuel_lbs=None) -> None:
        self.refuel = bool(refuel)
        self.device, self.dtype = device, dtype
        f0 = self.DEFAULT_LBS if fuel_lbs is None else fuel_lbs
        self.fuel = torch.as_tensor(f0, device=device, dtype=dtype).expand(n, 4).clone()
        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        self._r = t(self.TANK_IN)                              # (4, 3)
        self._cap = t(self.TANK_CAP_LBS)                       # (4,)
        self._cg_e = t(self.CG_EMPTY_IN)
        self._pm_r = [t(r) for _, r in self.POINTMASSES]
        self._w_fixed = self.EMPTY_LBS + sum(w for w, _ in self.POINTMASSES)
        self._mom_fixed = t([self.EMPTY_LBS * self.CG_EMPTY_IN[i]
                             + sum(w * r[i] for w, r in self.POINTMASSES)
                             for i in range(3)])
        self._axis = t((-1.0 / 12.0, 1.0 / 12.0, -1.0 / 12.0))  # 구조축 [in] -> body [ft]

    def _pm(self, m_slug: torch.Tensor, r_in: torch.Tensor, cg_in: torch.Tensor):
        """점질량의 (ixx, iyy, izz, ixz) 기여.  `FGMassBalance::GetPointmassInertia`."""
        v = (r_in - cg_in) * self._axis
        vx, vy, vz = v.unbind(-1)
        return (m_slug * (vy * vy + vz * vz), m_slug * (vx * vx + vz * vz),
                m_slug * (vx * vx + vy * vy), -m_slug * vx * vz)

    def props(self, fuel: torch.Tensor, cg_tank: torch.Tensor | None = None):
        """(MassProps[SI], cg_in [(N,3) in], mass_slugs [(N,)]).  `MassModel.props` 와 같은 꼴.

        `cg_tank` 는 탱크 평행축 항의 기준점이다 (직전 프레임 무게중심).  없으면
        지금 무게중심을 쓴다 (정지 상태 -- 트림 직후처럼 연료가 안 변한 경우).
        """
        w = self._w_fixed + fuel.sum(-1)
        cg = (self._mom_fixed + (fuel.unsqueeze(-1) * self._r).sum(-2)) / w.unsqueeze(-1)
        cgt = cg if cg_tank is None else cg_tank
        ixx, iyy, izz, ixz = (torch.full_like(w, v) for v in self.J_EMPTY)
        terms = [self._pm(torch.full_like(w, self.EMPTY_LBS / G_STD_FPS2), self._cg_e, cg)]
        for (pw, _), pr in zip(self.POINTMASSES, self._pm_r):
            terms.append(self._pm(torch.full_like(w, pw / G_STD_FPS2), pr, cg))
        for i in range(4):
            terms.append(self._pm(fuel[..., i] / G_STD_FPS2, self._r[i], cgt))
        for a, b, c, d in terms:
            ixx, iyy, izz, ixz = ixx + a, iyy + b, izz + c, ixz + d
        mass_slugs = w / G_STD_FPS2
        mp = RB.MassProps(mass=mass_slugs * SLUG, ixx=ixx * SLUGFT2,
                          iyy=iyy * SLUGFT2, izz=izz * SLUGFT2, ixz=ixz * SLUGFT2)
        return mp, cg, mass_slugs

    def advance(self, fuel: torch.Tensor, flow_pps: torch.Tensor) -> torch.Tensor:
        """한 물리 프레임: 소모 뒤 (켜져 있으면) 급유.  탱크 사이 이송·투하는 없다."""
        need = (flow_pps * DT_PHYS).unsqueeze(-1)
        has = fuel > 0.0
        # 개수(int64)는 연료 dtype 으로 바꿔 나눈다 -- 파이썬 실수/정수 텐서는 float32 로 승격된다.
        per = need / has.sum(-1, keepdim=True).clamp_min(1).to(fuel.dtype)
        left = fuel - per
        fuel = torch.where(has, torch.where(left >= 0.0, left, torch.zeros_like(left)), fuel)
        if self.refuel:
            open_ = (100.0 * fuel / self._cap) < 99.99
            k = open_.sum(-1, keepdim=True).clamp_min(1).to(fuel.dtype)
            add = torch.where(open_, (self.REFUEL_PPS * DT_PHYS) / k, torch.zeros_like(fuel))
            fuel = torch.minimum(fuel + add, self._cap)
        return fuel

    @classmethod
    def internal_first(cls, total_lbs: torch.Tensor) -> torch.Tensor:
        """연료 합계 [lb] -> 탱크 4 개.  내부 두 탱크를 먼저 채우고 넘치면 외부로."""
        cap_in = cls.TANK_CAP_LBS[0] + cls.TANK_CAP_LBS[1]
        t = total_lbs.clamp(0.0, sum(cls.TANK_CAP_LBS))
        inner = t.clamp(max=cap_in) / 2.0
        outer = (t - cap_in).clamp_min(0.0) / 2.0
        return torch.stack((inner, inner, outer, outer), -1)


# =============================================================================
# 트림 격자 -- JSBSim do_trim(1) 을 (고도 x 속도 x 연료) 로 잰 표
# =============================================================================

TRIM_TABLE = os.path.join(_HERE, "f16_trim.npz")
#: 표의 값 열 이름 (npz 키와 같다).
TRIM_KEYS = ("theta", "phi", "pitch_trim", "elevator_rad", "throttle", "n2")


class TrimGrid:
    """`f16_trim_build.py` 가 만든 표를 삼선형 보간한다 (그래프 안에서 돈다).

    축은 해면고도 [ft], 진대기속도 [kt], 연료 합계 [lb] (탱크 배치는
    `TankMass.internal_first`).  트림이 안 잡힌 칸은 `ok=False` 로 표시돼 있고,
    보간에 쓰이는 여덟 모서리 중 하나라도 그러면 결과의 `ok` 가 False 다
    (JSBSim `do_trim` 이 실패하는 곳과 같다 -- 값은 가장 가까운 잡힌 칸 것을 쓴다).
    """

    def __init__(self, path: str = TRIM_TABLE, device="cpu", dtype=torch.float32):
        z = np.load(path)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device, dtype=dtype)
        self.h_ft, self.v_kt, self.fuel_lbs = (t(z["h_ft"]), t(z["v_kt"]), t(z["fuel_lbs"]))
        self._axes_py = tuple(tuple(float(x) for x in z[k]) for k in ("h_ft", "v_kt", "fuel_lbs"))
        ok = np.asarray(z["ok"], dtype=bool)
        vals = np.stack([np.asarray(z[k], dtype=np.float64) for k in TRIM_KEYS], -1)
        if not ok.all():
            good = np.argwhere(ok)
            for bad in np.argwhere(~ok):
                j = good[np.abs(good - bad).sum(-1).argmin()]
                vals[tuple(bad)] = vals[tuple(j)]
        self.vals = t(vals)                                   # (H, V, F, K)
        self.ok = torch.as_tensor(ok, device=device)

    @staticmethod
    def _loc(x: torch.Tensor, axis: torch.Tensor, lo: float, hi: float):
        xc = x.clamp(lo, hi)
        i = torch.bucketize(xc, axis).clamp(1, axis.numel() - 1)
        x0, x1 = axis[i - 1], axis[i]
        return i, ((xc - x0) / (x1 - x0)).clamp(0.0, 1.0), (x >= lo) & (x <= hi)

    def __call__(self, h_ft: torch.Tensor, v_kt: torch.Tensor, fuel_lbs: torch.Tensor):
        """-> (dict 값 `(N,)`, ok `(N,)` bool)."""
        (hl, hh), (vl, vh), (fl, fh) = ((a[0], a[-1]) for a in self._axes_py)
        ih, wh, inh = self._loc(h_ft, self.h_ft, hl, hh)
        iv, wv, inv = self._loc(v_kt, self.v_kt, vl, vh)
        iff, wf, inf_ = self._loc(fuel_lbs, self.fuel_lbs, fl, fh)
        out = 0.0
        ok = inh & inv & inf_
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    w = ((wh if a else 1 - wh) * (wv if b else 1 - wv)
                         * (wf if c else 1 - wf)).unsqueeze(-1)
                    ii, jj, kk = ih - 1 + a, iv - 1 + b, iff - 1 + c
                    out = out + w * self.vals[ii, jj, kk]
                    ok = ok & self.ok[ii, jj, kk]
        return dict(zip(TRIM_KEYS, out.unbind(-1))), ok


# =============================================================================
# 조종간 직접 입력 플랜트
# =============================================================================

class F16Stick:
    """F-16 `N` 대를 조종간 네 개로 물리 프레임(1/120 s)씩 전진시키는 배치 플랜트.

        dyn = F16Stick(4096, device="cuda")
        ok = dyn.reset(pos_ned, psi, vt_ms)            # 수평 트림으로 시작
        for _ in range(1200):
            dyn.step(stick)                            # (N, 4) = aileron, elevator, rudder, throttle
        s = dyn.state()                                # pos_ned, quat, euler, uvw, pqr, ...

    **스틱 규약** ("+ = 오른쪽 / 당김"): aileron·elevator·rudder 는 [-1, 1],
    throttle 은 [0, 1] (1 = 애프터버너 최대).  JSBSim `fcs/*-cmd-norm` 으로는
    (aileron, -elevator, -rudder, throttle) 이다 (`BatchFLCS.from_stick`).

    **`step(u)` 한 번 = JSBSim 의 `set_controls(u); run()` 한 번.**  JSBSim 은
    프레임을 적분(Propagate)으로 시작하므로 `u` 가 만든 힘은 **다음** 프레임의
    적분에 들어간다.  그래서 이 플랜트도 직전 스틱을 들고 있다가 한 프레임 늦게
    쓴다 (`self.stick`).  리셋 직후 들고 있는 스틱은 트림 조종값이다.  상태
    (위치·자세·속도·각속도)는 JSBSim 의 `run()` 직후와 같은 시점이다.

    한 프레임 안의 계산 순서는 JSBSim 모델 실행 순서 그대로다 (`_substep`).
    `fdm_verify.py` 가 JSBSim 한가운데 상태를 심고 한 프레임씩 전부 대조한다.
    """

    _RB_STATE = ("pos_ned", "uvw", "quat", "pqr", "_fresh")

    def __init__(self, n: int, device="cuda", dtype=torch.float32, *,
                 lat0_deg: float = 0.0, mass: str = "tanks", refuel: bool = False,
                 fuel_lbs=None, gravity_ms2: float | None = None,
                 integrator: str = "jsbsim", tables: str | None = None,
                 trim_table: str | None = TRIM_TABLE) -> None:
        self.n = self.N = int(n)
        self.device = torch.device(device)
        self.dtype = dtype
        self.lat0_deg = float(lat0_deg)
        dev = str(self.device)
        tab = tables or os.path.join(_HERE, "f16_tables.npz")
        g = gravity_j2_ms2(lat0_deg, RB.H_REF / FT) if gravity_ms2 is None else gravity_ms2
        self.rb = RB.RigidBody6DOF((self.N,), dt=DT_PHYS, device=self.device,
                                   dtype=dtype, integrator=integrator,
                                   lat0_deg=lat0_deg, gravity=g)
        self.flcs = BatchFLCS(self.N, device=dev, dtype=dtype, dt=DT_PHYS)
        self.aero = F16Aero(tab, device=dev, dtype=dtype)
        self.turbine = F100Turbine(tab, device=dev, dtype=dtype)
        self.atmos = StdAtmosphere(device=dev, dtype=dtype)
        if mass == "tanks":
            self.mass = TankMass(self.N, self.device, dtype, refuel=refuel, fuel_lbs=fuel_lbs)
        else:
            raise ValueError(f"mass={mass!r}")
        self.trim = (TrimGrid(trim_table, device=self.device, dtype=dtype)
                     if trim_table and os.path.exists(trim_table) else None)

        self._rp = tuple(float(v) for v in self.aero.aerorp_in)
        self.rb.mass_props = self.mass.props(self.mass.fuel)[0]

        z = torch.zeros(self.N, device=self.device, dtype=dtype)
        self.n2 = z.clone()
        self.n2norm = z.clone()
        self.fuel = self.mass.fuel
        self.ff_pps = z.clone()
        self.pitch_trim = z.clone()
        self.a_body = torch.zeros(self.N, 3, device=self.device, dtype=dtype)
        self.wdot_i = torch.zeros(self.N, 3, device=self.device, dtype=dtype)
        self.n_pilot = torch.zeros(self.N, 3, device=self.device, dtype=dtype)
        self.n_pilot[:, 2] = -1.0
        #: JSBSim 이 지금 들고 있는 스틱 (다음 프레임의 힘에 쓰인다).
        self.stick = torch.zeros(self.N, 4, device=self.device, dtype=dtype)
        #: 엔진 스냅 대기 (리셋 뒤 두 번째 프레임) / 트림 프레임 (리셋 뒤 첫 프레임).
        self._engine_trim = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self._trim_frame = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        #: 탱크 평행축 항의 기준 무게중심 (직전 프레임) [in].  탱크 모델에서만 쓴다.
        self._tank = isinstance(self.mass, TankMass)
        self._cg_tank = self.mass.props(self.mass.fuel)[1].clone()


    # -- 리셋 -----------------------------------------------------------------

    def _mask(self, mask):
        return (torch.ones(self.N, dtype=torch.bool, device=self.device)
                if mask is None else mask.reshape(self.N).to(device=self.device,
                                                             dtype=torch.bool))

    def _seed_common(self, m0, alpha0, ptrim, ele_rad, n2_0, v_ned, h_ft, speed,
                     stick0, fuel0) -> None:
        """강체를 뺀 나머지 상태(FLCS 지연버퍼, 엔진, 연료, 직전 비력·스틱)를 트림 값으로 민다."""
        m1 = m0.unsqueeze(-1)

        def put(buf, val):
            buf.copy_(torch.where(m1 if buf.ndim == 2 else m0, val, buf))

        _, P_psf, _, a_fps = self.atmos(h_ft)
        mach0 = (speed / FT) / a_fps
        vc0 = vcalibrated_kts(mach0, P_psf)
        vg0 = torch.hypot(v_ned[..., 0], v_ned[..., 1]) / FT
        tef = tef_control_seed(vc0, mach0)
        self.flcs.reset(mask=m0, alpha_rad=alpha0,
                        n_pilot_z_norm=torch.full_like(speed, -1.0),
                        mach=mach0, vc_kts=vc0, vg_fps=vg0,
                        ele_pos_norm=ele_rad / ELEVATOR_RANGE_RAD,
                        tef_control=tef)
        put(self.pitch_trim, ptrim)
        put(self.n2, n2_0)
        self._engine_trim.logical_or_(m0)
        self._trim_frame.logical_or_(m0)
        self.n2norm.copy_((self.n2 - self.turbine.idle_n2) / self.turbine.n2_factor)
        put(self.fuel, fuel0)
        if self._tank:
            put(self._cg_tank, self.mass.props(fuel0)[1])
        put(self.ff_pps, torch.zeros_like(speed))
        ct, st_ = torch.cos(alpha0), torch.sin(alpha0)
        a_trim = torch.stack((G_STD_FPS2 * st_, torch.zeros_like(ct),
                              -G_STD_FPS2 * ct), dim=-1)
        put(self.a_body, a_trim)
        put(self.wdot_i, torch.zeros_like(self.wdot_i))
        npil = torch.zeros_like(self.n_pilot)
        npil[:, 0], npil[:, 2] = st_, -ct
        put(self.n_pilot, npil)
        put(self.stick, stick0)


    def reset(self, pos_ned: torch.Tensor, psi: torch.Tensor, vt_ms: torch.Tensor,
              fuel_lbs: torch.Tensor | None = None,
              mask: torch.Tensor | None = None) -> torch.Tensor:
        """수평 트림 비행으로 시작한다 (JSBSim `do_trim(1)` 과 같은 자리).

        pos_ned  (N, 3) [m]   북·동·아래.  아래 = -해면고도.
        psi      (N,)   [rad] 진방위 (북 0, 동 +pi/2).
        vt_ms    (N,)   [m/s] 진대기속도.
        fuel_lbs (N, 4) [lb]  탱크별 연료.  없으면 f16.xml 기본 (1,500, 1,500, 0, 0).
        mask     (N,)   bool  켜진 기체만 리셋 (그래프 안에서 돈다).

        돌려주는 `(N,)` bool 은 트림 표 안(트림이 잡히는 영역)인지다.
        """
        if self.trim is None:
            raise RuntimeError("트림 표(f16_trim.npz)가 없다 -- f16_trim_build.py 를 돌려라")
        if not isinstance(self.mass, TankMass):
            raise RuntimeError("reset() 은 탱크 질량 모델 전용이다")
        dt, dev = self.dtype, self.device
        pos = pos_ned.to(device=dev, dtype=dt).reshape(self.N, 3)
        psi = psi.to(device=dev, dtype=dt).reshape(self.N)
        speed = vt_ms.to(device=dev, dtype=dt).reshape(self.N)
        fuel0 = (torch.as_tensor(TankMass.DEFAULT_LBS, device=dev, dtype=dt).expand(self.N, 4)
                 if fuel_lbs is None else fuel_lbs.to(device=dev, dtype=dt).reshape(self.N, 4))
        h_ft = -pos[:, 2] / FT
        tr, ok = self.trim(h_ft, speed / KT_TO_MS, fuel0.sum(-1))
        quat = RB.quat_from_euler(tr["phi"], tr["theta"], psi)
        v_ned = torch.stack((speed * torch.cos(psi), speed * torch.sin(psi),
                             torch.zeros_like(speed)), -1)
        uvw = torch.einsum("...ij,...j->...i", RB.dcm_l2b(quat), v_ned)
        alpha0 = torch.atan2(uvw[:, 2], uvw[:, 0])
        m0 = self._mask(mask)
        m1 = m0.unsqueeze(-1)
        rb = self.rb
        torch.where(m1, pos, rb.pos_ned, out=rb.pos_ned)
        torch.where(m1, quat, rb.quat, out=rb.quat)
        torch.where(m1, uvw, rb.uvw, out=rb.uvw)
        rb.pqr.mul_(~m1)
        rb._fresh |= m1
        stick0 = torch.zeros_like(self.stick)
        stick0[:, 3] = tr["throttle"]
        self._seed_common(m0, alpha0, tr["pitch_trim"], tr["elevator_rad"], tr["n2"],
                          v_ned, h_ft, speed, stick0, fuel0)
        return ok

    # -- 전진 -----------------------------------------------------------------

    def step(self, stick: torch.Tensor, substeps: int = 1) -> None:
        """`substeps` 번의 `set_controls(stick); run()`.  `stick` 은 (N, 4)."""
        stick = stick.to(device=self.device, dtype=self.dtype).reshape(self.N, 4)
        carry = (self.n_pilot, self.n2, self.n2norm, self.fuel,
                 self.a_body, self.wdot_i, self.ff_pps)
        held = self.stick
        for _ in range(int(substeps)):
            carry = self._substep(*carry, stick=held)
            held = stick
        for buf, val in zip((self.n_pilot, self.n2, self.n2norm, self.fuel,
                             self.a_body, self.wdot_i, self.ff_pps), carry):
            buf.copy_(val)
        self.stick.copy_(stick)

    def _substep(self, n_pilot, n2, n2norm, fuel, a_body, wdot_i, ff_pps, *, stick):
        """물리 한 프레임.  JSBSim 모델 실행 순서: 질량 -> 보조량 -> FCS -> 추진 -> 공력 -> 적분."""
        rb = self.rb
        if self._tank:
            mp, cg_in, mass_slugs = self.mass.props(fuel, self._cg_tank)
            self._cg_tank.copy_(cg_in)
        else:
            mp, cg_in, mass_slugs = self.mass.props(fuel)
        pos, uvw, quat, pqr = rb.pos_ned, rb.uvw, rb.quat, rb.pqr
        u, v, w = uvw.unbind(-1)
        p, q, r = pqr.unbind(-1)
        phi, theta, psi = RB.euler_from_quat(quat)

        # --- 보조량 (FGAuxiliary) -------------------------------------------
        h_m = -pos[..., 2]
        h_ft = h_m / FT
        vt_ms = torch.linalg.vector_norm(uvw, dim=-1).clamp_min(1e-6)
        vt_fps = vt_ms / FT
        alpha = torch.atan2(w, u)
        beta = torch.atan2(v, torch.sqrt(u * u + w * w).clamp_min(1e-9))
        _, P_psf, rho, a_fps = self.atmos(h_ft)
        mach = vt_fps / a_fps
        qbar = 0.5 * rho * vt_fps * vt_fps
        vc_kts = vcalibrated_kts(mach, P_psf)
        v_ned = torch.einsum("...ji,...j->...i", RB.dcm_l2b(quat), uvw)
        vg_fps = torch.hypot(v_ned[..., 0], v_ned[..., 1]) / FT

        w_i = pqr + torch.einsum("...ij,...j->...i", RB.dcm_l2b(quat),
                                 rb.w_earth_ned.expand_as(pqr))
        n_pilot = pilot_accel_g(a_body, wdot_i, w_i, eye_arm_ft(cg_in))
        npz, npy = n_pilot[..., 2], n_pilot[..., 1]

        # --- FLCS: 스틱 -> 조종면 (한 프레임 지연은 BatchFLCS 가 스스로 한다) ----
        ail_u, ele_u, rud_u, thr_u = stick.unbind(-1)
        ail_c, ele_c, rud_c, thr_c = BatchFLCS.from_stick(
            ail_u, ele_u, rud_u, thr_u)
        fl = self.flcs.step(ail_c, ele_c, rud_c, thr_c,
                            alpha_rad=alpha, p_aero=p, q_aero=q, r_aero=r,
                            n_pilot_z_norm=npz, n_pilot_y_norm=npy,
                            mach=mach, vc_kts=vc_kts, vg_fps=vg_fps,
                            theta_rad=theta, phi_rad=phi,
                            pitch_trim_cmd=self.pitch_trim)

        # --- 추진 (`fcs/throttle-cmd-norm` 을 넣는다 -- 2 배는 터빈이 곱한다) ----
        pos_cmd, aug_cmd = F100Turbine.split_throttle(thr_c)
        thrust, n2, n2norm = self.turbine.step(
            n2, n2norm, thr_c, mach, h_ft, DT_PHYS, rho=rho)
        # 트림 뒤 첫 "진짜" 프레임(리셋 뒤 두 번째)은 N2 를 목표로 스냅한다
        # (JSBSim tpTrim 한 프레임).  스냅이 없는 기체는 where 가 값을 그대로 둔다.
        snap = self._engine_trim & ~self._trim_frame
        n2 = torch.where(snap, self.turbine.idle_n2
                         + pos_cmd * self.turbine.n2_factor, n2)
        n2norm = torch.where(snap, pos_cmd, n2norm)
        thrust = torch.where(
            snap, self.turbine.thrust(pos_cmd, aug_cmd, mach, h_ft), thrust)
        ff_pps = torch.where(
            snap, fuel_flow_pps(thrust, aug_cmd, n2norm), ff_pps)
        self._engine_trim.logical_and_(~snap)
        self._trim_frame.zero_()

        # --- 공력 -------------------------------------------------------------
        st = AeroState(alpha=alpha, beta=beta, mach=mach, qbar=qbar, vt=vt_fps,
                       p=p, q=q, r=r,
                       elevator=fl.elevator_pos_rad,
                       aileron=fl.aileron_pos_rad,
                       rudder=fl.rudder_pos_rad,
                       lef=fl.lef_pos_rad,
                       speedbrake=fl.speedbrake_pos_rad,
                       flaperon_mix=fl.flaperon_mix_rad,
                       gear=fl.gear_pos_norm)
        out = self.aero(st)
        l_cg, m_cg, n_cg = self._moments_about_cg(out, cg_in)

        # --- 합력 -> 적분 -----------------------------------------------------
        fbx = out["fbx"] + thrust
        f_lbs = torch.stack((fbx, out["fby"], out["fbz"]), dim=-1)
        m_lbsft = torch.stack(
            (l_cg, m_cg + (cg_in[..., 2] / 12.0) * thrust, n_cg), dim=-1)
        wrench = RB.Wrench(force=f_lbs * LBF, moment=m_lbsft * LBFFT)
        wdot_i = RB._solve_inertia(mp, RB._moment_minus_gyro(mp, w_i,
                                                             wrench.moment))
        a_body = f_lbs / mass_slugs.unsqueeze(-1)
        rb.step(wrench, mp)
        ff_pps = fuel_flow_seek(ff_pps, fuel_flow_pps(thrust, aug_cmd, n2norm))
        fuel = self.mass.advance(fuel, ff_pps)
        return n_pilot, n2, n2norm, fuel, a_body, wdot_i, ff_pps

    def _moments_about_cg(self, out: dict, cg_in: torch.Tensor):
        """공력 모멘트를 기준점(AERORP)에서 무게중심으로 옮긴다."""
        rp = self._rp
        dx = (cg_in[..., 0] - rp[0]) / 12.0
        dy = (rp[1] - cg_in[..., 1]) / 12.0
        dz = (cg_in[..., 2] - rp[2]) / 12.0
        fx, fy, fz = out["fbx"], out["fby"], out["fbz"]
        return (out["l"] + dy * fz - dz * fy,
                out["m"] + dz * fx - dx * fz,
                out["n"] + dx * fy - dy * fx)

    # -- 읽기 -----------------------------------------------------------------

    def state(self) -> dict:
        """지금 상태 (복사본).  단위는 SI 와 라디안, 연료만 [lb]."""
        rb = self.rb
        phi, theta, psi = RB.euler_from_quat(rb.quat)
        u, v, w = rb.uvw.unbind(-1)
        return dict(pos_ned=rb.pos_ned.clone(), quat=rb.quat.clone(),
                    euler=torch.stack((phi, theta, psi), -1), uvw=rb.uvw.clone(),
                    pqr=rb.pqr.clone(), vel_ned=rb.velocity_ned(),
                    alpha=torch.atan2(w, u),
                    beta=torch.atan2(v, torch.sqrt(u * u + w * w).clamp_min(1e-9)),
                    n2=self.n2.clone(), fuel_lbs=self.fuel.clone(),
                    n_pilot=self.n_pilot.clone())

    def watched_tensors(self) -> dict:
        """스텝을 건너 사는 상태 전부 (CUDA 그래프에서 주소가 안 바뀌어야 한다)."""
        out = {f"rb.{k}": getattr(self.rb, k) for k in self._RB_STATE}
        for key, lst in getattr(self.rb, "_hist", {}).items():
            for i, t in enumerate(lst):
                out[f"rb._hist.{key}[{i}]"] = t
        out.update({f"flcs.{k}": getattr(self.flcs, k) for k in BatchFLCS.N_STATE_KEYS})
        out.update({f"plant.{k}": getattr(self, k) for k in
                    ("n2", "n2norm", "fuel", "ff_pps", "pitch_trim", "a_body",
                     "wdot_i", "n_pilot", "stick", "_engine_trim", "_trim_frame",
                     "_cg_tank")})
        return out
