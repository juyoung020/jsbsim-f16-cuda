# SPDX-License-Identifier: GPL-3.0-or-later
"""F-16 공력 (JSBSim f16.xml `aerodynamics` 절) 의 torch 배치 재구현.

    from jsbsim_f16_cuda.aero import F16Aero, AeroState
    aero = F16Aero(device="cuda")
    out  = aero(state)          # 힘·모멘트, 전부 (B,) 텐서

무엇을 돌려주는가
-----------------
JSBSim 의 `aero/coefficient/*` 는 **이름과 달리 무차원 계수가 아니다.**  XML 의
product 안에 이미 `aero/qbar-psf * metrics/Sw-sqft` (모멘트축은 기준길이까지)
가 곱해져 있어, 값의 단위는 lbf / lbf*ft 다.  여기서도 같은 규약을 쓴다 --
JSBSim 프로퍼티와 1:1 로 대조하기 위해서다.  무차원 계수가 필요하면
`out["CD"]` 처럼 나눠 놓은 것을 쓴다.

축 규약 (여기서 조용히 틀린다)
------------------------------
f16.xml 의 축 이름이 DRAG/SIDE/LIFT 라 JSBSim 은 `atLiftDrag` 로 잡고,
`FGAerodynamics::Run()` 이 이렇게 옮긴다.

    vFb = Tw2b * ( -DRAG, +SIDE, -LIFT )

즉 DRAG 축의 합은 **뒤로 양(+)**, LIFT 축의 합은 **위로 양(+)**, SIDE 축의
합은 **오른쪽으로 양(+)** 이다.  Tw2b 는 alpha, beta 로 만든다.

    Tw2b = [[ ca*cb, -ca*sb, -sa],
            [    sb,     cb,   0],
            [ sa*cb, -sa*sb,  ca]]

주의: JSBSim 이 프로퍼티로 내놓는 `forces/fw{x,y,z}-aero-lbs` 는 부호를 뒤집기
**전**의 (DRAG, SIDE, LIFT) 다.  풍축 힘이라고 그대로 적분하면 항력이 추력이
된다.  여기서도 대조를 위해 out["fwx"] 는 JSBSim 규약을 따른다.

모멘트는 ROLL/PITCH/YAW 축 합이 **공력 기준점(AERORP) 둘레** 값이다.  무게중심
둘레로 옮기려면 `r x F` 를 더한다 (r = AERORP 를 동체축에서 본 위치) --
`arm_body_ft()` / `moments_about_cg()` 참고.  JSBSim 의 `moments/l-aero-lbsft`
는 이미 옮긴 값이다.

기준 길이
---------
ROLL/YAW 는 bw(30 ft), PITCH 는 cbar(11.32 ft).  둘을 바꿔 쓰면 모멘트가
2.65 배 틀리는데 부호도 모양도 그럴듯해서 안 보인다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .f16_interp import interp1_stack, interp2_stack, regrid1, regrid2

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "f16_tables.npz")

AXES = ("DRAG", "SIDE", "LIFT", "ROLL", "PITCH", "YAW")

#: 테이블이 읽는 프로퍼티 -> 우리 상태 이름.  metrics/* 는 상수라 빠져 있다.
PROP2STATE = {
    "aero/qbar-psf": "qbar",
    "aero/alpha-rad": "alpha",
    "aero/beta-rad": "beta",
    "aero/bi2vel": "bi2vel",
    "aero/ci2vel": "ci2vel",
    "aero/function/kCLge": "kCLge",
    "velocities/mach": "mach",
    "velocities/p-aero-rad_sec": "p",
    "velocities/q-aero-rad_sec": "q",
    "velocities/r-aero-rad_sec": "r",
    "fcs/aileron-pos-rad": "aileron",
    "fcs/elevator-pos-rad": "elevator",
    "fcs/rudder-pos-rad": "rudder",
    "fcs/lef-pos-rad": "lef",
    "fcs/speedbrake-pos-rad": "speedbrake",
    "fcs/flaperon-mix-rad": "flaperon_mix",
    "gear/gear-pos-norm": "gear",
    "aero/h_b-mac-ft": "h_b_mac",
}

METRIC_PROPS = {"metrics/Sw-sqft": "Sw_sqft",
                "metrics/bw-ft": "bw_ft",
                "metrics/cbarw-ft": "cbar_ft"}


@dataclass
class AeroState:
    """(B,) 텐서 묶음.  단위는 JSBSim 내부 단위 (rad, rad/s, psf, ft/s, ft).

    기본값은 **JSBSim 번들 f16 을 로드한 직후의 실측값**이지 "깨끗한 형상"이 아니다.

      gear = 1.0        착륙장치가 **내려가 있다** (`gear/gear-pos-norm` = 1.0).
                        CDgear 가 0.027 -- 순항 총항력의 **절반**이다.
                        0 으로 두면 기체가 조용히 훨씬 잘 난다.
      speedbrake = 0.0  실측 0.  FCS 가 스피드브레이크를 자동으로 여는 조건이
                        alpha>=53 deg **그리고** v<=18 fps 뿐이라 안 걸리고,
                        `fcs/speedbrake-cmd-norm` 을 쓰는 쪽도 없다.
                        (명령하면 `fcs/speedbrake-pos-rad` 는 제대로 움직인다 --
                        JSBSim 이 rad/deg/norm 세 쌍을 같이 굴린다.  그 경우도
                        대조해 뒀으니 공력 쪽은 어느 쪽이든 맞다.)

    `lef` 와 `flaperon_mix` 는 기본값 0 이지만 **비행 중엔 0 이 아니다.**
    반드시 flcs.py 가 준 값을 넣어라.

      lef           기어가 내려가 있어 스위치의 `gear==0 -> 0.436` 가지가 죽는다.
                    실측 고유값은 {-0.0349, 0.0, 0.262} 셋뿐이고
                    (alpha>5 deg -> 0.262, alpha<5 deg 이면서 mach>0.9 -> -0.0349),
                    **alpha 15 deg 를 넘겨도 0.436 은 나오지 않는다.**
      flaperon_mix  실측 [-0.90, 0.0].  롤 채널의 플래퍼론 믹스라 에일러론을
                    쓰는 동안 계속 0 이 아니다.
    """
    alpha: torch.Tensor
    beta: torch.Tensor
    mach: torch.Tensor
    qbar: torch.Tensor                       # psf
    vt: torch.Tensor                         # ft/s  (bi2vel, ci2vel 용)
    p: torch.Tensor
    q: torch.Tensor
    r: torch.Tensor
    elevator: torch.Tensor                   # rad
    aileron: torch.Tensor
    rudder: torch.Tensor
    lef: torch.Tensor = None
    speedbrake: torch.Tensor = None
    flaperon_mix: torch.Tensor = None
    gear: torch.Tensor = None
    h_b_mac: torch.Tensor = None             # 지면효과용.  None 이면 지면효과 없음

    def fill(self):
        z = torch.zeros_like(self.alpha)
        for name in ("lef", "speedbrake", "flaperon_mix"):
            if getattr(self, name) is None:
                setattr(self, name, z)
        if self.gear is None:
            self.gear = z + 1.0             # 실측: 번들 f16 은 기어를 내린 채 시작한다
        if self.h_b_mac is None:
            self.h_b_mac = z + 100.0        # 표가 1.1 에서 물리므로 kCLge = 1
        return self


class F16Aero(torch.nn.Module):
    """f16.xml 의 공력 함수 36 개를 배치로 한 번에 푼다.

    빠른 이유는 세 가지다.
      1. 같은 독립변수를 쓰는 표를 합집합 격자로 다시 샘플링해 **하나로 쌓았다**.
         searchsorted 가 표마다가 아니라 독립변수마다 한 번만 돈다 (5 번).
      2. product 의 property 인자를 **gather 한 번**으로 모은다.  36 개 함수의
         곱셈이 (B, F, K) 텐서 하나의 prod 로 끝난다.
      3. 축 합산이 (B, F) x (F, 6) 행렬곱 하나다.
    파이썬 루프는 초기화 때만 돈다.  런타임 커널 수는 배치 크기와 무관하게 약 20.
    """

    def __init__(self, path: str = TABLES, device="cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        from .parse_f16 import unflatten
        with np.load(path, allow_pickle=False) as npz:
            model = unflatten(npz)

        self.metrics = {k: float(v) for k, v in model["metrics"].items()
                        if k != "aerorp_in"}
        self.aerorp_in = np.asarray(model["metrics"]["aerorp_in"], dtype=float)
        self.dtype = dtype

        fns = []                                       # (axis_idx, function)
        for ai, axis in enumerate(AXES):
            for fn in model["axes"][axis]:
                fns.append((ai, fn))
        # kCLge 는 다른 함수가 property 로 읽으므로 먼저 푼다.
        self.helpers = model["helpers"]
        self.fn_names = [fn["name"] for _, fn in fns]
        F = len(fns)

        # ---- 1) 표를 독립변수별로 묶어 합집합 격자로 다시 샘플링 --------------
        t1: dict[str, list] = {}                       # row_var -> [(fi, table)]
        t2: dict[tuple, list] = {}
        for fi, (_, fn) in enumerate(fns):
            for fac in fn["factors"]:
                if fac["kind"] != "table":
                    continue
                t = fac["table"]
                if int(t["ndim"]) == 1:
                    t1.setdefault(str(t["row_var"]), []).append((fi, t))
                else:
                    t2.setdefault((str(t["row_var"]), str(t["col_var"])), []).append((fi, t))

        self._g1 = []          # (state_name, bp tensor, data tensor (K,N), [fi...])
        self._g2 = []          # (row_state, col_state, rbp, cbp, data (K,R,C), [fi...])
        tv_of_fn = np.zeros(F, dtype=np.int64)         # 0 = 표 없음(1.0)
        slot = 1
        for var, items in t1.items():
            bp = np.unique(np.concatenate([t["row"] for _, t in items]))
            data = np.stack([regrid1(t["row"], t["data"], bp) for _, t in items])
            idx = []
            for fi, _ in items:
                tv_of_fn[fi] = slot; idx.append(slot); slot += 1
            self._g1.append((PROP2STATE[var], bp, data, idx))
        for (rvar, cvar), items in t2.items():
            rbp = np.unique(np.concatenate([t["row"] for _, t in items]))
            cbp = np.unique(np.concatenate([t["col"] for _, t in items]))
            data = np.stack([regrid2(t["row"], t["col"], t["data"], rbp, cbp)
                             for _, t in items])
            idx = []
            for fi, _ in items:
                tv_of_fn[fi] = slot; idx.append(slot); slot += 1
            self._g2.append((PROP2STATE[rvar], PROP2STATE[cvar], rbp, cbp, data, idx))

        # ---- 2) property 인자 -> gather 인덱스, value/metrics -> 상수 ---------
        self.state_names = list(dict.fromkeys(PROP2STATE.values()))
        col_of = {n: i + 1 for i, n in enumerate(self.state_names)}   # 0 은 1.0
        prop_idx, const = [], np.ones(F)
        for fi, (_, fn) in enumerate(fns):
            cols = []
            for fac in fn["factors"]:
                if fac["kind"] == "value":
                    const[fi] *= float(fac["value"])
                elif fac["kind"] == "property":
                    p = str(fac["prop"])
                    if p in METRIC_PROPS:
                        const[fi] *= self.metrics[METRIC_PROPS[p]]
                    else:
                        cols.append(col_of[PROP2STATE[p]])
            prop_idx.append(cols)
        K = max(len(c) for c in prop_idx)
        gidx = np.zeros((F, K), dtype=np.int64)
        for fi, cols in enumerate(prop_idx):
            gidx[fi, :len(cols)] = cols

        # ---- 3) 축 합산 행렬 ---------------------------------------------------
        A = np.zeros((F, len(AXES)))
        for fi, (ai, _) in enumerate(fns):
            A[fi, ai] = 1.0

        # kCLge (지면효과) 는 단독 1D 표
        kge = self.helpers[0]
        assert kge["name"] == "aero/function/kCLge", kge["name"]
        kt = kge["factors"][0]["table"]

        reg = lambda n, a: self.register_buffer(n, torch.as_tensor(a), persistent=False)
        reg("_gather_idx", gidx)
        reg("_tv_of_fn", tv_of_fn)
        reg("_const", const.astype(np.float64))
        reg("_axis_mat", A.astype(np.float64))
        reg("_kge_bp", np.asarray(kt["row"]))
        reg("_kge_data", np.asarray(kt["data"])[None, :])
        for i, (_, bp, data, _) in enumerate(self._g1):
            reg(f"_g1bp{i}", bp); reg(f"_g1d{i}", data)
        for i, (_, _, rbp, cbp, data, _) in enumerate(self._g2):
            reg(f"_g2r{i}", rbp); reg(f"_g2c{i}", cbp); reg(f"_g2d{i}", data)

        self.to(device=device, dtype=dtype)
        # searchsorted 의 인덱스는 정수여야 하므로 dtype 변환에서 되돌린다
        self._gather_idx = self._gather_idx.long()
        self._tv_of_fn = self._tv_of_fn.long()

    # ------------------------------------------------------------------ 본체

    def kCLge(self, h_b_mac: torch.Tensor) -> torch.Tensor:
        return interp1_stack(self._kge_bp, self._kge_data, h_b_mac)[:, 0]

    def functions(self, s: AeroState) -> torch.Tensor:
        """36 개 함수값 (B, F).  JSBSim 의 aero/coefficient/* 와 같은 단위."""
        B = s.alpha.shape[0]
        vt = s.vt.clamp_min(1e-6)
        bi2vel = self.metrics["bw_ft"] / (2.0 * vt)
        ci2vel = self.metrics["cbar_ft"] / (2.0 * vt)
        kge = self.kCLge(s.h_b_mac)

        env = dict(qbar=s.qbar, alpha=s.alpha, beta=s.beta, bi2vel=bi2vel,
                   ci2vel=ci2vel, kCLge=kge, mach=s.mach, p=s.p, q=s.q, r=s.r,
                   aileron=s.aileron, elevator=s.elevator, rudder=s.rudder,
                   lef=s.lef, speedbrake=s.speedbrake,
                   flaperon_mix=s.flaperon_mix, gear=s.gear, h_b_mac=s.h_b_mac)

        ones = torch.ones(B, 1, dtype=s.alpha.dtype, device=s.alpha.device)
        X = torch.cat([ones] + [env[n].reshape(B, 1) for n in self.state_names], 1)

        tv = [ones]
        for i, (var, _, _, _) in enumerate(self._g1):
            tv.append(interp1_stack(getattr(self, f"_g1bp{i}"),
                                    getattr(self, f"_g1d{i}"), env[var]))
        for i, (rv, cv, _, _, _, _) in enumerate(self._g2):
            tv.append(interp2_stack(getattr(self, f"_g2r{i}"), getattr(self, f"_g2c{i}"),
                                    getattr(self, f"_g2d{i}"), env[rv], env[cv]))
        TV = torch.cat(tv, 1)

        base = X[:, self._gather_idx].prod(dim=2)           # (B, F)
        return base * TV[:, self._tv_of_fn] * self._const

    def forward(self, s: AeroState, want_functions: bool = False) -> dict:
        s.fill()
        vals = self.functions(s)
        ax = vals @ self._axis_mat                          # (B, 6)
        D, Y, L = ax[:, 0], ax[:, 1], ax[:, 2]
        l, m, n = ax[:, 3], ax[:, 4], ax[:, 5]

        ca, sa = torch.cos(s.alpha), torch.sin(s.alpha)
        cb, sb = torch.cos(s.beta), torch.sin(s.beta)
        # Tw2b 에 들어가는 것은 (-D, Y, -L) 이다.  JSBSim 이 프로퍼티로 내놓는
        # forces/fw{x,y,z}-aero-lbs 는 부호를 뒤집기 **전**의 (D, Y, L) 이라
        # 아래 out["fwx"] 는 그쪽 규약을 따른다 (실측으로 확인).
        nx, ny, nz = -D, Y, -L
        fbx = ca * cb * nx - ca * sb * ny - sa * nz
        fby = sb * nx + cb * ny
        fbz = sa * cb * nx - sa * sb * ny + ca * nz

        qS = (s.qbar * self.metrics["Sw_sqft"]).clamp_min(1e-9)
        out = dict(
            drag=D, side=Y, lift=L, l=l, m=m, n=n,
            fbx=fbx, fby=fby, fbz=fbz, fwx=D, fwy=Y, fwz=L,
            CD=D / qS, CY=Y / qS, CL=L / qS,
            Cl=l / (qS * self.metrics["bw_ft"]),
            Cm=m / (qS * self.metrics["cbar_ft"]),
            Cn=n / (qS * self.metrics["bw_ft"]),
        )
        if want_functions:
            out["functions"] = vals
        return out

    def arm_body_ft(self, cg_in):
        """AERORP 을 동체축에서 본 위치 = JSBSim `FGMassBalance::StructuralToBody`.

        구조축은 x 가 **뒤로**, y 가 오른쪽, z 가 **위로** 인 인치 좌표계고,
        동체축은 x 가 앞, y 가 오른쪽, z 가 아래다.  그래서 x 와 z 만 뒤집힌다.

            ( (cg_x - rp_x)/12,  (rp_y - cg_y)/12,  (cg_z - rp_z)/12 )   [ft]

        부호를 통째로 뒤집으면 모멘트 이동항의 부호가 반대가 되는데, 크기가
        맞아서 그래프로는 안 보인다 (실제로 한 번 틀렸다가 대조에서 잡혔다).
        """
        rp = self.aerorp_in
        cg = np.asarray(cg_in, dtype=float)
        return ((cg[..., 0] - rp[0]) / 12.0,
                (rp[1] - cg[..., 1]) / 12.0,
                (cg[..., 2] - rp[2]) / 12.0)

    def moments_about_cg(self, out: dict, cg_in) -> tuple:
        """AERORP 둘레 모멘트를 무게중심 둘레로 옮긴다.

        JSBSim `FGAerodynamics::Run()`:  vMoments = vDXYZcg x vForces + vMomentsMRC
        """
        dx, dy, dz = self.arm_body_ft(cg_in)
        fx, fy, fz = out["fbx"], out["fby"], out["fbz"]
        return (out["l"] + dy * fz - dz * fy,
                out["m"] + dz * fx - dx * fz,
                out["n"] + dx * fy - dy * fx)
