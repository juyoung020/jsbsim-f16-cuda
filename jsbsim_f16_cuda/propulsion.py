# SPDX-License-Identifier: GPL-3.0-or-later
"""F100-PW-229 터빈 (JSBSim `FGTurbine`) 의 torch 배치 재구현 + 표준대기.

JSBSim 원본 (`src/models/propulsion/FGTurbine.cpp::Run()`) 을 그대로 옮긴다.

    idlethrust = MilThrust * IdleThrust(mach, 밀도고도)
    milthrust  = (MilThrust - idlethrust) * MilThrust(mach, 밀도고도)
    N2        <- Seek(N2, IdleN2 + throttle*N2_factor, spoolup, spooldown)
    N2norm     = (N2 - IdleN2) / N2_factor
    thrust     = idlethrust + milthrust * N2norm^2
    AB (augmethod 2):  thrust += (MaxThrust*AugThrust(mach,고도) - thrust) * AugCmd

스풀은 1 차 지연이 아니라 **속도 제한(rate limit)** 이다 (`Seek`).  램프로
올라가다 목표에 닿으면 딱 멈춘다.  지수 수렴으로 바꾸면 전투 중 가감속 응답이
통째로 달라진다.

스로틀 규약 (JSBSim 실측)
---------------------------------------------
f16.xml Throttle 채널이 `throttle-pos-norm = 2 x throttle-cmd-norm` 이다.
FGTurbine 은 pos > 1.0 이면 넘치는 만큼을 AugmentCmd 로 떼어 낸다.

    cmd 0.0 -> pos 0.0 -> idle
    cmd 0.5 -> pos 1.0 -> military (17,800 lbf 기준)
    cmd 1.0 -> pos 2.0 -> full afterburner (AugmentCmd 1.0)

밀도고도
--------
표준대기 + 무풍이면 밀도고도 = 기하고도다(실측으로 확인).  대기 모델에 온도
편차를 넣게 되면 여기가 조용히 틀린다.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .f16_interp import interp2_stack, regrid2

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "f16_tables.npz")

# ------------------------------------------------------------------ 표준대기
# 1976 US Standard Atmosphere.  JSBSim FGStandardAtmosphere 의 온도표와 같다.
_H_GEOPOT = np.array([0.0, 36089.2388, 65616.7979, 104986.8766,
                      154199.4751, 170603.6751, 200131.2337, 259186.3517])
_T_BASE = np.array([518.67, 389.97, 389.97, 411.57, 487.17, 487.17, 454.77, 325.17])
_P_SL = 2116.228            # psf
# R 과 g0 는 JSBSim 에서 역산한 값이다 (추측하지 않는다).
#   P/(rho T) 가 고도와 무관하게 1716.55716,  대류권 지수 5.25589425 -> g/R
_R_ENG = 1716.55716         # ft2/(s2 R)
_G0 = 32.174049             # ft/s2
_RHO_SL = _P_SL / (_R_ENG * 518.67)    # slug/ft3.  JSBSim 값과 9e-6 차이
_GAMMA = 1.4
_EARTH_R = 20855531.5       # ft  (JSBSim FGAtmosphere::EarthRadius)


def _std_layers():
    """층마다 (h_base, T_base, lapse, P_base) 를 미리 적분해 둔다."""
    lapse = np.zeros(len(_H_GEOPOT))
    for i in range(len(_H_GEOPOT) - 1):
        lapse[i] = (_T_BASE[i + 1] - _T_BASE[i]) / (_H_GEOPOT[i + 1] - _H_GEOPOT[i])
    P = np.empty(len(_H_GEOPOT)); P[0] = _P_SL
    for i in range(len(_H_GEOPOT) - 1):
        dh = _H_GEOPOT[i + 1] - _H_GEOPOT[i]
        if abs(lapse[i]) < 1e-12:
            P[i + 1] = P[i] * np.exp(-_G0 * dh / (_R_ENG * _T_BASE[i]))
        else:
            P[i + 1] = P[i] * (_T_BASE[i + 1] / _T_BASE[i]) ** (-_G0 / (_R_ENG * lapse[i]))
    return lapse, P


_LAPSE, _P_BASE = _std_layers()


class StdAtmosphere(torch.nn.Module):
    """h [ft MSL] -> T [R], P [psf], rho [slug/ft3], a [ft/s].  배치."""

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        for n, a in (("h0", _H_GEOPOT), ("T0", _T_BASE),
                     ("lam", _LAPSE), ("P0", _P_BASE)):
            self.register_buffer(n, torch.as_tensor(a), persistent=False)
        self.to(device=device, dtype=dtype)

    def forward(self, h_ft: torch.Tensor):
        hg = h_ft * _EARTH_R / (_EARTH_R + h_ft)          # 기하 -> 지오퍼텐셜
        i = (torch.searchsorted(self.h0, hg.contiguous().detach(), right=True) - 1)
        i = i.clamp_(0, self.h0.numel() - 1)
        h0, T0, lam, P0 = self.h0[i], self.T0[i], self.lam[i], self.P0[i]
        dh = hg - h0
        T = T0 + lam * dh
        iso = lam.abs() < 1e-12
        P = torch.where(iso,
                        P0 * torch.exp(-_G0 * dh / (_R_ENG * T0)),
                        P0 * (T / T0) ** (-_G0 / (_R_ENG * torch.where(iso, torch.ones_like(lam), lam))))
        rho = P / (_R_ENG * T)
        a = torch.sqrt(_GAMMA * _R_ENG * T)
        return T, P, rho, a


# --------------------------------------------------------------------- 터빈

class F100Turbine(torch.nn.Module):
    """배치 엔진.  상태는 (B,) 짜리 N2 하나(와 N2norm)다.

        eng = F100Turbine(device="cuda")
        N2 = eng.init_n2(throttle_cmd)          # do_trim 직후의 JSBSim 과 같다
        thrust, N2, N2norm = eng.step(N2, N2norm, throttle_cmd, mach, h_ft, dt)
    """

    def __init__(self, path: str = TABLES, device="cpu", dtype=torch.float32):
        super().__init__()
        from .parse_f16 import unflatten
        with np.load(path, allow_pickle=False) as npz:
            model = unflatten(npz)
        e = model["engine"]
        self.mil = float(e["milthrust"])
        self.max = float(e["maxthrust"])
        self.bpr = float(e["bypassratio"])
        self.idle_n2 = float(e["idlen2"])
        self.n2_factor = float(e["maxn2"]) - self.idle_n2
        # FGSpoolUp: delay = factor * 90 / (BPR + 3);  N2 는 up 1.0, down 3.0
        self.spool_up = 1.0 * 90.0 / (self.bpr + 3.0)
        self.spool_dn = 3.0 * 90.0 / (self.bpr + 3.0)

        t = e["tables"]
        order = ["IdleThrust", "MilThrust", "AugThrust"]
        rbp = np.unique(np.concatenate([t[k]["row"] for k in order]))     # mach
        cbp = np.unique(np.concatenate([t[k]["col"] for k in order]))     # 밀도고도
        data = np.stack([regrid2(t[k]["row"], t[k]["col"], t[k]["data"], rbp, cbp)
                         for k in order])
        self.register_buffer("_rbp", torch.as_tensor(rbp), persistent=False)
        self.register_buffer("_cbp", torch.as_tensor(cbp), persistent=False)
        self.register_buffer("_data", torch.as_tensor(data), persistent=False)
        self.atmos = StdAtmosphere(device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)

    # -- 스로틀 -------------------------------------------------------------

    @staticmethod
    def split_throttle(cmd: torch.Tensor):
        """throttle-cmd-norm -> (ThrottlePos 0..1, AugmentCmd 0..1)."""
        pos = 2.0 * cmd
        aug = (pos - 1.0).clamp(0.0, 1.0)
        return pos.clamp(0.0, 1.0), aug

    def init_n2(self, cmd: torch.Tensor) -> torch.Tensor:
        """JSBSim 의 tpTrim 분기와 같다 -- 스풀을 거치지 않고 목표에 바로 앉힌다."""
        thr, _ = self.split_throttle(cmd)
        return self.idle_n2 + thr * self.n2_factor

    # -- 한 스텝 -------------------------------------------------------------

    def step(self, N2, N2norm, cmd, mach, h_ft, dt: float, rho=None):
        thr, aug = self.split_throttle(cmd)
        if rho is None:
            _, _, rho, _ = self.atmos(h_ft)
        density_ratio = rho / _RHO_SL

        # FGSpoolUp::GetValue -- 직전 스텝의 N2norm 을 본다 (JSBSim 과 같은 순서)
        n = (N2norm + 0.1).clamp_max(1.0)
        denom = 1.0 + 3.0 * (1.0 - n) ** 3 + (1.0 - density_ratio)
        up = self.spool_up / denom
        dn = self.spool_dn / denom

        target = self.idle_n2 + thr * self.n2_factor
        rise = (N2 < target)
        step = torch.where(rise, dt * up, -dt * dn)
        N2 = torch.where(rise, torch.minimum(N2 + step, target),
                         torch.maximum(N2 + step, target))
        N2norm = (N2 - self.idle_n2) / self.n2_factor

        thrust = self.thrust(N2norm, aug, mach, h_ft)
        return thrust, N2, N2norm

    def thrust(self, N2norm, aug_cmd, mach, density_alt_ft):
        tv = interp2_stack(self._rbp, self._cbp, self._data, mach, density_alt_ft)
        idle = self.mil * tv[:, 0]
        milt = (self.mil - idle) * tv[:, 1]
        t = idle + milt * N2norm * N2norm
        tdiff = self.max * tv[:, 2] - t
        return torch.where(aug_cmd > 0.0, t + tdiff * aug_cmd.clamp_max(1.0), t)
