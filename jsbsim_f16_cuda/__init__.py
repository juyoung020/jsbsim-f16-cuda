# SPDX-License-Identifier: GPL-3.0-or-later
"""JSBSim F-16 비행모델(f16.xml + F100 엔진 + FLCS)을 PyTorch 로 옮긴 배치 시뮬레이터.

    from jsbsim_f16_cuda import F16Stick
    dyn = F16Stick(4096, device="cuda")

JSBSim 엔진 전체가 아니라 **F-16 모델 하나**다.  다른 기체 XML 은 돌지 않는다.
"""
from .f16_core import F16Stick, TankMass, TrimGrid, gravity_j2_ms2  # noqa: F401

__version__ = "0.1.0"
