# SPDX-License-Identifier: GPL-3.0-or-later
"""JSBSim F-16 기준기 -- `jsbsim` 패키지(1.3.0)를 직접 세팅한다.

GPU 포팅을 대조할 **정답**이다.  JSBSim 프로퍼티만 쓴다.  기본은 JSBSim 표준 동작이다:
트림 뒤 조종은 트림값 그대로, 연료는 정상 소모, 급유 없음 (`refuel=True` 로 켠다).

JSBSim 을 조용히 틀리게 만드는 함정 다섯 개를 전부 막는다:

    1. `run_ic()` 도 `do_trim()` 도 엔진을 켜지 않는다 -> `set-running = 1`.
    2. `run_ic()` 는 FCS 적분기를 안 지운다 -> 스틱 0 뒤 `reset_to_initial_conditions(0)`.
       그 호출은 탱크도 XML 기본값으로 되돌리므로 **연료는 그 뒤에** 넣는다.
    3. 자세·각속도 초기조건을 전부 명시한다 (안 하면 직전 비행의 자세가 남는다).
    4. 위치는 `position/distance-from-start-*` 가 아니라 위경도에서 직접 평면으로 편다.
       위도는 **측지** 쪽을 쓰고 읽는다 (지심과 섞으면 0.19 도 = 20 km 어긋난다).
    5. `FGFDMExec` 의 루트 디렉터리를 명시한다 (다른 JSBSim 설치가 끼어든다).

조종 규약 (`set_controls`) -- "+ = 오른쪽 / 당김" 이다:

    fcs/aileron-cmd-norm  = clip(aileron, -1, 1)
    fcs/elevator-cmd-norm = -clip(elevator, -1, 1)     음수가 기수 올림이라 뒤집는다
    fcs/rudder-cmd-norm   = -clip(rudder, -1, 1)       양수가 왼쪽 요라 뒤집는다
    fcs/throttle-cmd-norm = clip(throttle, 0, 1)       1.0 = 애프터버너 최대
"""
from __future__ import annotations

import math
import os

import jsbsim

R_EARTH = 6371000.0
FT = 0.3048
N_TANKS = 4
#: f16.xml 기본 탱크량 [lb] (내부 1,500 x 2, 외부 0 x 2).
DEFAULT_FUEL_LBS = (1500.0, 1500.0, 0.0, 0.0)


def _clip(u: float, lo: float, hi: float) -> float:
    return min(max(float(u), lo), hi)


class JSBSimF16Ref:
    """JSBSim F-16 한 대.  `reset` -> (`set_controls` -> `run_one`) x N."""

    def __init__(self, dt_physics: float = 1.0 / 120.0, h0_ft: float = 20000.0,
                 lat0_deg: float = 0.0, lon0_deg: float = 0.0,
                 refuel: bool = False, post_trim_controls=None,
                 zero_external_flow: bool = False) -> None:
        self._dt = float(dt_physics)
        self._h0_ft = float(h0_ft)
        self.lat0_deg, self.lon0_deg = float(lat0_deg), float(lon0_deg)
        self.refuel = bool(refuel)
        self.post_trim_controls = post_trim_controls
        self.zero_external_flow = bool(zero_external_flow)
        self._fdm = jsbsim.FGFDMExec(os.path.dirname(jsbsim.__file__), None)
        self._fdm.set_debug_level(0)
        self._fdm.load_model("f16")
        self._fdm.set_dt(self._dt)
        self._lat0 = math.radians(self.lat0_deg)
        self._lon0 = math.radians(self.lon0_deg)
        self.trim_ok = False


    @property
    def fdm(self):
        return self._fdm

    @property
    def dt_physics(self) -> float:
        return self._dt

    def reset(self, x: float = 0.0, y: float = 0.0, psi: float = 0.0,
              v_kt: float = 400.0, h_ft: float | None = None,
              fuel_lbs=None) -> bool:
        """수평 트림으로 시작한다.  `x` 동쪽 · `y` 북쪽 [m] (평면 원점은 lat0/lon0),
        `psi` 진방위 [rad], `v_kt` 진대기속도, `h_ft` 해면고도, `fuel_lbs` 탱크 4 개 [lb].

        돌려주는 값은 트림 수렴 여부다 (저속·고고도에서는 수평비행이 안 된다).
        """
        h_ft = self._h0_ft if h_ft is None else float(h_ft)
        fdm = self._fdm
        lat = self.lat0_deg + math.degrees(y / R_EARTH)
        lon = self.lon0_deg + math.degrees(x / (R_EARTH * math.cos(self._lat0)))
        fdm["ic/h-sl-ft"] = h_ft
        fdm["ic/vt-kts"] = v_kt
        fdm["ic/psi-true-deg"] = math.degrees(psi) % 360.0
        fdm["ic/lat-geod-deg"] = lat
        fdm["ic/long-gc-deg"] = lon
        for k in ("phi", "theta", "alpha", "beta", "gamma"):
            fdm[f"ic/{k}-deg"] = 0.0
        for k in "pqr":
            fdm[f"ic/{k}-rad_sec"] = 0.0
        self.set_controls(0.0, 0.0, 0.0, 0.0)
        fdm.run_ic()
        fdm.reset_to_initial_conditions(0)
        if fuel_lbs is not None:
            for i, c in enumerate(fuel_lbs):
                fdm[f"propulsion/tank[{i}]/contents-lbs"] = float(c)
        fdm["propulsion/engine[0]/set-running"] = 1
        fdm["fcs/throttle-cmd-norm"] = 1.0
        try:
            fdm.do_trim(1)
            self.trim_ok = True
        except RuntimeError:
            self.trim_ok = False
        if self.post_trim_controls is not None:
            self.set_controls(*self.post_trim_controls)
        if self.zero_external_flow:
            for tank in range(int(fdm["propulsion/total-fuel-lbs"] > 0) * 8):
                try:
                    fdm[f"propulsion/tank[{tank}]/external-flow-rate-pps"] = 0.0
                except Exception:
                    break
        if self.refuel:
            fdm["propulsion/refuel"] = 1
        return self.trim_ok

    def set_controls(self, aileron: float, elevator: float, rudder: float,
                     throttle: float) -> None:
        fdm = self._fdm
        fdm["fcs/aileron-cmd-norm"] = _clip(aileron, -1.0, 1.0)
        fdm["fcs/elevator-cmd-norm"] = -_clip(elevator, -1.0, 1.0)
        fdm["fcs/rudder-cmd-norm"] = -_clip(rudder, -1.0, 1.0)
        fdm["fcs/throttle-cmd-norm"] = _clip(throttle, 0.0, 1.0)

    def run_one(self) -> None:
        self._fdm.run()

    def tanks(self) -> tuple:
        return tuple(float(self._fdm[f"propulsion/tank[{i}]/contents-lbs"])
                     for i in range(N_TANKS))

    def xy(self) -> tuple:
        """평면 위치 (동, 북) [m] -- `reset` 의 x, y 와 같은 평면."""
        lat = math.radians(self._fdm["position/lat-geod-deg"])
        lon = self._fdm["position/long-gc-rad"]
        return ((lon - self._lon0) * R_EARTH * math.cos(self._lat0),
                (lat - self._lat0) * R_EARTH)


