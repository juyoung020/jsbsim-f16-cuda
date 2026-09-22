# SPDX-License-Identifier: GPL-3.0-or-later
"""6 자유도 강체 운동방정식 + 적분기.  전부 배치 텐서, 파이썬 루프 없음.

공력·추력 계수는 여기 없다 -- 이 모듈은 **body 축 힘[N]·모멘트[N*m] 를 받아 상태를
굴리는 일만** 한다.  계수 쪽과의 경계는 `Wrench` 하나다.

--------------------------------------------------------------------------
좌표계
--------------------------------------------------------------------------
내부 상태는 전부 NED/body 다::

    pos_ned   위치 [m]      북, 동, 아래 (아래 = -해면고도)
    uvw       body 속도 [m/s]
    quat      Tl2b 쿼터니언 (local NED -> body), 성분 순서 (w, x, y, z) -- JSBSim 과 같다
    pqr       body 각속도 [rad/s] (국소 NED 기준)

자세는 3-2-1 오일러 (psi -> theta -> phi).  편의용으로 ENU 9 값 형식
`[x 동, y 북, h 고도, phi, theta, psi, vx 동, vy 북, vz 위]` 로 넣고 빼는 함수
(`set_from_contract`, `reset_masked`, `to_contract`)도 둔다.

--------------------------------------------------------------------------
JSBSim 1.3.0 실측으로 확정한 사실
--------------------------------------------------------------------------
1. 적분기 (JSBSim 기본값, dt = 1/120 s)::

       rate/rotational       RectEuler        (pqr)
       rate/translational    AdamsBashforth2  (**관성계 속도**, u,v,w 가 아니다)
       position/rotational   RectEuler        (쿼터니언)
       position/translational AdamsBashforth3 (위치)

   병진 속도는 **동체축이 아니라 관성계(국소 NED)에서** 적분한다 (`FGPropagate::Run`
   이 관성 속도를 `vUVWidot` 으로 적분하고, 적분이 끝난 **새 자세**로 `vUVW` 를 다시
   만든다).  동체축에서 AB2 를 돌리면 연속시간에서는 같은 식인데 이산 다단계법에서는
   다르다 -- 과거 미분값이 **그때의 동체축** 좌표라, 기체가 도는 동안 그 사이
   회전(omega*dt)만큼 틀린 방향으로 외삽된다.  빠르게 롤하는 비행에서 JSBSim 자기
   기록으로 재면 동체축 AB2 는 프레임당 최대 0.10 ft/s 어긋나고, 국소 NED 의 AB2 는
   2.4e-4 ft/s 로 맞는다 (남는 것은 평평한 지구의 수송률 V/R).

   그래서 `integrator="jsbsim"` 이 기본이다.  RK4 가 "더 정확"하지만 맞춰야 할
   기준은 참값이 아니라 **JSBSim 궤적**이다.

2. `forces/fb*-total-lbs` 는 **중력을 포함하지 않는다.**  중력은 여기서 더한다.

3. 관성텐서의 비대각 성분은 `inertia/ixz-slugs_ft2` **값을 그대로** J[0,2] 에
   넣는다.  `f16.xml` 의 `negated_crossproduct_inertia="true"` 에 속아 부호를
   뒤집으면 각가속도 잔차가 1,700 배 튄다.  XML 의 ixx=9496 은 **빈 기체값**이고
   연료·조종사를 얹은 값은 다르다 (`f16_core.TankMass`).

4. 각가속도는 `J^-1 (M - w_i x J w_i)` 이고 `w_i` 는 **관성계 기준** 각속도
   = pqr + 지구자전분이다.

--------------------------------------------------------------------------
지구 모델 -- 평평한 지구 + 자전
--------------------------------------------------------------------------
위치는 기준 위도 `lat0_deg` 의 접평면에 둔다.  빼면 안 되는 두 항:

  * 코리올리 `-(pqr + 2*Omega_b) x uvw`.  **계수 2** 를 빠뜨리면 절반만 들어간다
    (450 kt 에서 v_dot 이 0.034 ft/s^2 틀린다 -- v_dot 전체의 26 배).
  * 자전 원심가속도의 연직 몫 Omega^2 R cos^2(lat) (g 의 약 0.2 %).  수평 몫은
    측지 연직 중력에 이미 흡수돼 있으므로 넣지 않는다 (넣으면 u_dot 이 나빠진다).

중력 크기는 `gravity` 로 받는다 (기본 `G0` = 위도 37.5665, 20,000 ft 의 JSBSim 값).
`f16_core.F16Stick` 은 자기 위도의 JSBSim J2 중력(`gravity_j2_ms2`)을 넘긴다.
고도에 따라서는 역제곱으로 늘인다.  뺀 것: 수송률(V/R), 지구 곡률에 의한 평면 왜곡.

--------------------------------------------------------------------------
질량
--------------------------------------------------------------------------
질량·관성은 **상태가 아니라 매 스텝 주입받는 입력**이다 (`step(wrench, mass_props)`).
연료가 줄거나 늘면 바뀌기 때문이다.  배치 텐서를 제자리로 갱신해 넘기면 CUDA
그래프와도 맞는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# -- 단위 --------------------------------------------------------------------
FT = 0.3048
LBF = 4.4482216152605          # lbf -> N
LBFFT = 1.3558179483314004     # lbf*ft -> N*m
SLUG = 14.593902937206364      # slug -> kg
SLUGFT2 = 1.3558179483314004   # slug*ft^2 -> kg*m^2

#: 20,000 ft 에서의 중력 가속도 [m/s^2].  JSBSim 실측 32.15939 ft/s^2.
G0 = 32.1593872843476 * FT
#: 위 G0 를 잰 기준 고도 [m] (20,000 ft).  역제곱 중력의 기준점이다.
H_REF = 20000.0 * FT
#: 지구 반지름 [m] (역제곱 중력·원심가속도의 r).
R_EARTH = 6371000.0
#: 지구 자전 각속도 [rad/s]
OMEGA_EARTH = 7.2921151467e-5
#: 위도를 안 주면 쓰는 기준 위도 [deg].  `G0` 가 이 위도의 값이다.
CENTER_LAT_DEG = 37.5665

#: JSBSim 이 트림 직후 보고하는 값 (runtime `inertia/*`, 기본 연료 3,000 lbs 포함).
#: `MassProps.f16_default` 가 쓴다.  XML 의 emptywt/ixx 가 아니다 -- 위 문서 3 번.
F16_MASS_SLUGS = 641.1999931994883
F16_IXX_SLUGFT2 = 12288.752252685077
F16_IYY_SLUGFT2 = 57107.52277243735
F16_IZZ_SLUGFT2 = 67072.31271410483
F16_IXZ_SLUGFT2 = -1059.8589065378305

#: ENU 9 값 형식의 채널 이름.
CONTRACT = ("x", "y", "h", "phi", "theta", "psi", "vx", "vy", "vz")


# ---------------------------------------------------------------------------
# 계수 모듈과의 경계
# ---------------------------------------------------------------------------
@dataclass
class Wrench:
    """공력·추력 모듈이 돌려줘야 하는 것.  **body 축, SI, 중력 제외.**

    body 축 = x 기수 앞, y 오른쪽 날개, z 아래 (JSBSim 과 동일).

    force  (..., 3)  [N]     공력 + 추력.  **중력을 넣지 마라** -- 여기서 더한다.
    moment (..., 3)  [N*m]   CG 기준 (L, M, N) = (롤 오른쪽+, 피치 기수위+, 요 오른쪽+)

    JSBSim 대응 프로퍼티::
        force  = (forces/fbx-total-lbs, fby, fbz) * LBF
        moment = (moments/l-total-lbsft, m, n) * LBFFT
    """

    force: Tensor
    moment: Tensor


@dataclass
class MassProps:
    """질량·관성.  스칼라(float)거나 배치 텐서 (...,) 둘 다 된다.

    관성은 **body 축 관성텐서 성분 그대로** 다.  `ixz` 는 JSBSim
    `inertia/ixz-slugs_ft2` 의 부호 그대로 J[0,2] = J[2,0] 에 들어간다.

    다섯 항목은 `(*batch,)` 텐서여야 한다 (기체마다 다른 값).  질량은 연료에 따라
    변하므로 연료 모델(`f16_core.TankMass`)이 매 스텝 갱신해서 넘긴다.  제자리
    갱신(`mp.mass.copy_(...)`)이면 그래프 캡처와도 호환된다.
    """

    mass: Tensor | float          # [kg]
    ixx: Tensor | float           # [kg*m^2]
    iyy: Tensor | float
    izz: Tensor | float
    ixz: Tensor | float

    def to(self, device, dtype, batch_shape: tuple[int, ...] = ()) -> "MassProps":
        """다섯 항목을 전부 `(*batch_shape,)` 디바이스 텐서로 만든다."""
        def cvt(v):
            t = torch.as_tensor(v, device=device, dtype=dtype)
            return t.expand(batch_shape).contiguous() if t.ndim == 0 else t.to(device, dtype)

        return MassProps(cvt(self.mass), cvt(self.ixx), cvt(self.iyy),
                         cvt(self.izz), cvt(self.ixz))

    @staticmethod
    def f16_default(device=None, dtype=torch.float32,
                    batch_shape: tuple[int, ...] = ()) -> "MassProps":
        """트림 직후 JSBSim 이 보고하는 값 (연료 3,000 lbs 포함)."""
        return MassProps(
            mass=F16_MASS_SLUGS * SLUG,
            ixx=F16_IXX_SLUGFT2 * SLUGFT2,
            iyy=F16_IYY_SLUGFT2 * SLUGFT2,
            izz=F16_IZZ_SLUGFT2 * SLUGFT2,
            ixz=F16_IXZ_SLUGFT2 * SLUGFT2,
        ).to(device or "cpu", dtype, batch_shape)


# ---------------------------------------------------------------------------
# 쿼터니언 (Tl2b: local NED -> body, 성분 순서 w,x,y,z)
# ---------------------------------------------------------------------------
def quat_from_euler(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    """3-2-1 오일러 -> Tl2b 쿼터니언 (..., 4)."""
    cp, sp = torch.cos(phi * 0.5), torch.sin(phi * 0.5)
    ct, st = torch.cos(theta * 0.5), torch.sin(theta * 0.5)
    cy, sy = torch.cos(psi * 0.5), torch.sin(psi * 0.5)
    return torch.stack(
        (
            cp * ct * cy + sp * st * sy,
            sp * ct * cy - cp * st * sy,
            cp * st * cy + sp * ct * sy,
            cp * ct * sy - sp * st * cy,
        ),
        dim=-1,
    )


def dcm_l2b(q: Tensor) -> Tensor:
    """Tl2b 행렬 (..., 3, 3).  v_body = Tl2b @ v_ned.

    JSBSim `FGQuaternion::ComputeDerived` 와 같은 식이다.
    """
    w, x, y, z = q.unbind(-1)
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    return torch.stack(
        (
            torch.stack((w2 + x2 - y2 - z2, 2 * (x * y + w * z), 2 * (x * z - w * y)), -1),
            torch.stack((2 * (x * y - w * z), w2 - x2 + y2 - z2, 2 * (y * z + w * x)), -1),
            torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x), w2 - x2 - y2 + z2), -1),
        ),
        dim=-2,
    )


def euler_from_quat(q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Tl2b 쿼터니언 -> (phi, theta, psi) [rad].  theta 는 +-pi/2 로 잘린다."""
    w, x, y, z = q.unbind(-1)
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    m02 = 2 * (x * z - w * y)
    phi = torch.atan2(2 * (y * z + w * x), w2 - x2 - y2 + z2)
    theta = torch.asin(torch.clamp(-m02, -1.0, 1.0))
    psi = torch.atan2(2 * (x * y + w * z), w2 + x2 - y2 - z2)
    return phi, theta, psi


def quat_dot(q: Tensor, pqr: Tensor) -> Tensor:
    """qdot = 0.5 * Omega(pqr) q.  JSBSim `FGQuaternion::GetQDot` 와 동일."""
    w, x, y, z = q.unbind(-1)
    p, qq, r = pqr.unbind(-1)
    return 0.5 * torch.stack(
        (
            -(x * p + y * qq + z * r),
            w * p + y * r - z * qq,
            w * qq + z * p - x * r,
            w * r + x * qq - y * p,
        ),
        dim=-1,
    )


def quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
class RigidBody6DOF:
    """배치 6-DOF 강체.  판마다 도는 파이썬 루프 없음.

    상태 (전부 `(*batch, 3)` 또는 `(*batch, 4)`)::

        pos_ned   위치 [m]     북, 동, **아래** (down = -고도MSL)
        uvw       body 속도 [m/s]
        quat      Tl2b 쿼터니언 (w, x, y, z)
        pqr       body 각속도 [rad/s]  (국소 NED 기준)

    `batch` 는 자유롭다 (예: `(N,)` 기체 N 대, `(B, 2)` 두 대씩 B 묶음).
    """

    #: `jsbsim` 모드가 필요로 하는 과거 미분값 개수 (AB3)
    _HIST = 3

    def __init__(
        self,
        batch_shape: tuple[int, ...],
        *,
        dt: float = 1.0 / 120.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        integrator: str = "jsbsim",
        mass_props: MassProps | None = None,
        earth_rotation: bool = True,
        frame_correction: bool = True,
        gravity_varies: bool = True,
        lat0_deg: float = CENTER_LAT_DEG,
        gravity: float = G0,
    ) -> None:
        if integrator not in ("jsbsim", "euler", "rk4"):
            raise ValueError(f"integrator={integrator!r}")
        self.batch_shape = tuple(batch_shape)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.integrator = integrator
        self.gravity = float(gravity)
        # 질량·관성은 **항상 디바이스 텐서로** 들고 있는다.  파이썬 실수로 두면
        # 매 스텝 호스트->디바이스 복사가 생겨 CUDA 그래프 캡처가 거부된다.
        self.mass_props = (mass_props or MassProps.f16_default()).to(
            self.device, dtype, self.batch_shape)

        z3 = lambda n=3: torch.zeros(*self.batch_shape, n, device=self.device, dtype=dtype)
        self.pos_ned = z3()
        self.uvw = z3()
        self.pqr = z3()
        self.quat = z3(4)
        self.quat[..., 0] = 1.0

        # 중력 [NED].  down 성분이 +g 다.
        self.g_ned = torch.zeros(3, device=self.device, dtype=dtype)
        self.g_ned[2] = self.gravity

        # 지구자전 각속도 [NED].  위도만으로 정해지는 상수 벡터다.
        self.w_earth_ned = torch.zeros(3, device=self.device, dtype=dtype)
        # 지구자전 **원심가속도** -Omega x (Omega x r) 의 계수 [1/s^2].
        # `derivatives` 에서 r = R_EARTH + h 를 곱해 쓴다.
        #
        # 연직 성분만 넣는다.  JSBSim 은 `gtStandard` 중력(지심 방향 GM/r^2)에
        # 이 항을 통째로 더하는데, 우리 `g_ned` 는 이미 **측지 연직**(local down)
        # 을 향하고 있어서 원심력의 수평(북) 성분은 그 기울기에 이미 흡수돼
        # 있다.  실측으로도 북 성분을 같이 넣으면 udot 오차가 1.5e-3 ->
        # 5.4e-2 ft/s^2 로 **36 배 나빠진다** (프레임 단위 JSBSim 대조).
        self.a_cent_ned = torch.zeros(3, device=self.device, dtype=dtype)
        if earth_rotation:
            lat = math.radians(lat0_deg)
            self.w_earth_ned[0] = OMEGA_EARTH * math.cos(lat)
            self.w_earth_ned[2] = -OMEGA_EARTH * math.sin(lat)
            self.a_cent_ned[2] = -(OMEGA_EARTH ** 2) * math.cos(lat) ** 2
        self._earth_rotation = bool(earth_rotation)
        self._frame_correction = bool(earth_rotation and frame_correction)
        self._gravity_varies = bool(gravity_varies)

        # Adams-Bashforth 과거 미분값.  **미리 잡아 두고 제자리로만 갱신한다.**
        # (3, *batch, 3) -- [0] 이 가장 최근.
        #
        # 🔴 `"uvw"` 칸에 든 것은 **국소 NED 좌표의 속도 미분**(지구 기준 속도의
        # 시간미분)이다 -- 동체축 `uvw_dot` 이 아니다.  JSBSim 이 관성계에서 속도를
        # 적분하기 때문이다 (머리말 1).  이름은 붙어 있는 곳(융합 커널 인자,
        # `fdm_verify.seed`)이 많아 그대로 뒀다.
        self._hist = {
            k: torch.zeros(self._HIST, *self.batch_shape, 3,
                           device=self.device, dtype=dtype)
            for k in ("uvw", "pos")
        }
        # "이 판은 방금 리셋됐으니 AB 이력을 현재 미분값으로 다시 채워라" 표시.
        # 디바이스 쪽 불리언이다 -- 호스트로 내려오면 그래프 캡처가 깨진다.
        self._fresh = torch.ones(*self.batch_shape, 1, device=self.device,
                                 dtype=torch.bool)

    # -- 경계 변환 ----------------------------------------------------------
    @staticmethod
    def _contract_to_internal(s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ENU 9 값 -> (pos_ned, quat, uvw).  상태를 건드리지 않는 순수 함수."""
        x, y, h, phi, theta, psi, vx, vy, vz = s.unbind(-1)
        pos_ned = torch.stack((y, x, -h), dim=-1)          # ENU -> NED: (y, x, -h)
        quat = quat_from_euler(phi, theta, psi)
        v_ned = torch.stack((vy, vx, -vz), dim=-1)
        uvw = torch.einsum("...ij,...j->...i", dcm_l2b(quat), v_ned)
        return pos_ned, quat, uvw

    def set_from_contract(self, s: Tensor) -> None:
        """ENU 9 값 `(*batch, 9)` = [x,y,h,phi,theta,psi,vx,vy,vz] 로 **전체**를 놓는다.

        속도는 ENU 라 body 로 돌려 넣는다.  **옆미끄럼각·받음각은 이 9 개에서
        복원된다** -- 속도 벡터 전체가 들어 있기 때문에 정보 손실이 없다.

        주의: 9 값에는 **각속도(p,q,r)가 없다.**  여기서는 0 으로 놓는다 --
        트림된 새 비행의 시작이라는 뜻이다.  비행 도중에 이걸 부르면 각속도가
        조용히 지워진다.  부분 리셋은 `reset_masked` 를 써라.
        """
        pos, quat, uvw = self._contract_to_internal(
            s.to(device=self.device, dtype=self.dtype))
        self.pos_ned.copy_(pos)
        self.quat.copy_(quat)
        self.uvw.copy_(uvw)
        self.pqr.zero_()
        self._fresh.fill_(True)

    def reset_masked(self, mask: Tensor, s: Tensor) -> None:
        """마스크가 켜진 기체만 ENU 9 값 `s` 로 되돌린다.  나머지는 그대로 둔다.

        `mask`  `(*batch,)` 또는 `(*batch, 1)` 불리언 **디바이스 텐서**.
        `s`     `(*batch, 9)` ENU 9 값.  마스크가 꺼진 자리 값은 무시된다
                (아무 값이나 넣어도 되지만 NaN 은 넣지 마라 -- `where` 는
                양쪽을 다 계산하므로 NaN 이 섞이면 전파될 수 있다).

        **`nonzero()`/불리언 인덱싱을 쓰지 않는다.**  둘 다 호스트 동기화라
        CUDA 그래프 캡처가 깨진다.  전부 `torch.where` 와 제자리 복사다.

        Adams-Bashforth 이력도 같이 무효화한다 (`_fresh`).  이걸 빼먹으면
        리셋된 판이 **이전 에피소드의 미분값**으로 처음 두 스텝을 굴려서,
        재현 안 되는 미세한 초기 오차가 에피소드마다 섞여 들어간다.
        """
        if mask.ndim == len(self.batch_shape):
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=self.device, dtype=torch.bool)
        pos, quat, uvw = self._contract_to_internal(
            s.to(device=self.device, dtype=self.dtype))
        # `out=` 로 제자리 기록한다.  `x.copy_(torch.where(...))` 보다 커널이
        # 절반이고, 무엇보다 `self.x = ...` 재대입이 아니라서 그래프 안전하다.
        torch.where(mask, pos, self.pos_ned, out=self.pos_ned)
        torch.where(mask, quat, self.quat, out=self.quat)
        torch.where(mask, uvw, self.uvw, out=self.uvw)
        self.pqr.mul_(~mask)                       # 리셋된 판의 각속도만 0
        self._fresh |= mask

    def to_contract(self) -> Tensor:
        """`(*batch, 9)` ENU 9 값으로 내보낸다."""
        phi, theta, psi = euler_from_quat(self.quat)
        v_ned = torch.einsum("...ji,...j->...i", dcm_l2b(self.quat), self.uvw)
        n, e, d = self.pos_ned.unbind(-1)
        vn, ve, vd = v_ned.unbind(-1)
        return torch.stack((e, n, -d, phi, theta, psi, ve, vn, -vd), dim=-1)

    def velocity_ned(self) -> Tensor:
        return torch.einsum("...ji,...j->...i", dcm_l2b(self.quat), self.uvw)

    # -- 운동방정식 ----------------------------------------------------------
    def derivatives(
        self,
        pos_ned: Tensor,
        uvw: Tensor,
        quat: Tensor,
        pqr: Tensor,
        wrench: Wrench,
        mp: MassProps,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """상태와 외력 -> (pos_dot, uvw_dot, quat_dot, pqr_dot).

        선가속도 (body 축)::

            uvw_dot = F/m + Tl2b*(g_ned + a_cent_ned) - (pqr + 2*Omega_b) x uvw

        **자전 각속도가 두 번 들어간다.**  JSBSim `FGAccelerations::
        CalculateUVWdot` 이 `-(vPQR + 2*vOmegaPlanet) x vUVW` 이기 때문이다 --
        하나는 body 프레임이 도는 몫(원심), 하나는 코리올리(2*Omega x V) 몫이다.
        한 번만 넣으면 450 kt 에서 v_dot 이 0.034 ft/s^2 (전체의 26 배) 틀린다.
        각가속도 쪽 `w_i` 는 한 번만 들어가는 것이 맞다 (아래).

        `a_cent_ned` 는 자전 원심가속도의 연직 성분이다.  빼면 w_dot 이
        0.070 ft/s^2 (= Omega^2 R cos^2(lat), g 의 0.22 %) 틀린다.

        실측 (프레임 단위 JSBSim 대조, 13 입력열 x 4 비행조건, float64,
        한 스텝 |오차| ft/s^2):

            항목      고치기 전            고친 뒤
            u_dot     5e-4 ~ 4e-3          6e-7 ~ 2e-4
            v_dot     2e-2 ~ 5e-2          8e-7 ~ 5e-5
            w_dot     6.6e-2 ~ 7.7e-2      1.8e-4 ~ 3.1e-4

        각가속도.  **우리 `pqr` 은 JSBSim 의 `velocities/p-rad_sec` 과 같은
        뜻이다 -- 지구와 함께 도는 프레임 기준**이지 관성계 기준이 아니다.
        관성계 기준 각가속도를 구한 뒤 프레임 회전분을 되돌려 준다::

            w_i      = pqr + Tl2b * w_earth
            w_i_dot  = J^-1 (M - w_i x J w_i)
            pqr_dot  = w_i_dot + pqr x (Tl2b * w_earth)

        마지막 항을 빼먹으면 |pqr|*Omega ~ 1.5e-4 rad/s^2 만큼 틀린다.
        """
        Tl2b = dcm_l2b(quat)
        w_e_b = torch.einsum("...ij,...j->...i", Tl2b, self.w_earth_ned.expand_as(pqr))
        w_i = pqr + w_e_b

        g_ned = self.g_ned.expand_as(uvw)
        if self._gravity_varies:
            # g(h) = g_ref * ((R + h_ref) / (R + h))^2.  고도가 고정이면 상수와
            # 같지만, 루프/줌클라임으로 +-8,000 ft 를 오가면 이게 60 s 에
            # 속도 1 m/s 를 만든다 (실측).  JSBSim 은 WGS84 중력을 쓴다.
            h = -pos_ned[..., 2:3]
            scale = ((R_EARTH + H_REF) / (R_EARTH + h)) ** 2
            g_ned = g_ned * scale
        if self._earth_rotation:
            # -Omega x (Omega x r) 의 연직 몫.  r = R_EARTH + h 이고
            # `-pos_ned[...,2]` 가 곧 h 다.
            g_ned = g_ned + self.a_cent_ned * (R_EARTH - pos_ned[..., 2:3])
        g_b = torch.einsum("...ij,...j->...i", Tl2b, g_ned)
        # mass 는 `(*batch,)` 텐서거나 스칼라.  둘 다 복사 없이 나눠진다 --
        # 여기서 `torch.as_tensor(..., device=)` 를 부르면 스텝마다 호스트->
        # 디바이스 복사가 생겨 그래프 캡처가 거부된다.
        mass = mp.mass
        if torch.is_tensor(mass) and mass.ndim:
            mass = mass.unsqueeze(-1)
        # 코리올리 항의 각속도는 `pqr + 2*Omega_b` 다 (`w_i` 가 아니다).
        w_cor = w_i + w_e_b
        uvw_dot = wrench.force / mass + g_b - torch.cross(w_cor, uvw, dim=-1)

        pqr_dot = _solve_inertia(mp, _moment_minus_gyro(mp, w_i, wrench.moment))
        if self._frame_correction:
            pqr_dot = pqr_dot + torch.cross(pqr, w_e_b, dim=-1)

        # 위치 미분은 NED 속도.  Tl2b^T @ uvw
        pos_dot = torch.einsum("...ji,...j->...i", Tl2b, uvw)
        return pos_dot, uvw_dot, quat_dot(quat, pqr), pqr_dot

    # -- 적분 ---------------------------------------------------------------
    def step(self, wrench: Wrench, mass_props: MassProps | None = None) -> None:
        """한 스텝(dt) 전진.  `wrench` 는 **현재 상태에서 평가된** 힘이다.

        상태 텐서는 **재대입하지 않고 제자리로만** 갱신한다 (`copy_`/`add_`).
        `self.x = torch.where(...)` 로 새 텐서를 매달면 CUDA 그래프에 캡처된
        커널은 캡처 당시 주소를 계속 읽어서 **그래프 모드에서만 조용히 틀린다.**
        """
        mp = mass_props or self.mass_props
        if self.integrator == "rk4":
            self._step_rk4(wrench, mp)
        else:
            self._step_linear_multistep(wrench, mp)
        # AB 이력을 다시 채웠으니 표시를 내린다.
        self._fresh.zero_()

    def _step_linear_multistep(self, wrench: Wrench, mp: MassProps) -> None:
        dt = self.dt
        pos_dot, uvw_dot, q_dot, pqr_dot = self.derivatives(
            self.pos_ned, self.uvw, self.quat, self.pqr, wrench, mp)

        if self.integrator == "euler":
            self.pqr.add_(pqr_dot, alpha=dt)
            self.quat.add_(q_dot, alpha=dt)
            self._renorm_quat()
            self.uvw.add_(uvw_dot, alpha=dt)
            self.pos_ned.add_(pos_dot, alpha=dt)
            return

        # "jsbsim" 모드.  JSBSim 1.3.0 의 기본 조합을 그대로 흉내낸다:
        #   pqr  RectEuler / quat RectEuler / 속도 AB2 (관성계) / pos AB3
        # 네 갈래의 미분값은 **전부 스텝 시작 상태에서** 한 번에 뽑는다
        # (위 `derivatives` 호출).  즉 자세는 갱신 전 pqr 로 굴러간다 --
        # 갱신된 pqr 을 쓰도록 바꾸면 JSBSim 과의 자세 오차가 커진다 (실측).
        #
        # 속도는 **국소 NED 에서** 적분하고 새 자세로 동체축에 되돌린다 (머리말 1).
        #   a_ned = Tb2l (uvw_dot + pqr x uvw)       <- 지구 기준 속도의 NED 미분
        #   v_ned' = v_ned + dt (1.5 a_ned[k] - 0.5 a_ned[k-1])
        #   uvw'  = Tl2b(q') v_ned'                   <- **적분한 뒤의** 자세
        # `pos_dot` 이 곧 v_ned (= Tb2l uvw) 다.
        Tl2b = dcm_l2b(self.quat)
        a_ned = torch.einsum("...ji,...j->...i", Tl2b,
                             uvw_dot + torch.cross(self.pqr, self.uvw, dim=-1))
        vel_h = self._push("uvw", a_ned)
        pos_h = self._push("pos", pos_dot)

        self.pqr.add_(pqr_dot, alpha=dt)
        self.quat.add_(q_dot, alpha=dt)
        self._renorm_quat()
        v_ned = pos_dot + dt * (1.5 * vel_h[0] - 0.5 * vel_h[1])
        self.uvw.copy_(torch.einsum("...ij,...j->...i", dcm_l2b(self.quat), v_ned))
        self.pos_ned.add_(
            (23.0 / 12.0) * pos_h[0] - (16.0 / 12.0) * pos_h[1] + (5.0 / 12.0) * pos_h[2],
            alpha=dt)

    def _renorm_quat(self) -> None:
        self.quat.div_(self.quat.norm(dim=-1, keepdim=True).clamp_min(1e-12))

    def _step_rk4(self, wrench: Wrench, mp: MassProps) -> None:
        """고전 RK4.  외력은 스텝 동안 고정으로 본다.

        JSBSim 도 한 프레임 동안 힘을 고정하므로 이것이 공정한 비교다.  RK4 가
        더 정확히 푸는 것은 **운동학 비선형성**(w x v, 쿼터니언 회전)이다.
        """
        dt = self.dt
        s0 = (self.pos_ned, self.uvw, self.quat, self.pqr)

        def f(s):
            return self.derivatives(*s, wrench, mp)

        def add(s, d, a):
            return tuple(si + a * di for si, di in zip(s, d))

        k1 = f(s0)
        k2 = f(add(s0, k1, dt * 0.5))
        k3 = f(add(s0, k2, dt * 0.5))
        k4 = f(add(s0, k3, dt))
        for buf, a, b, c, d in zip(s0, k1, k2, k3, k4):
            buf.add_(a + 2 * b + 2 * c + d, alpha=dt / 6.0)
        self._renorm_quat()

    def _push(self, key: str, val: Tensor) -> Tensor:
        """AB 과거값 버퍼를 한 칸 밀고 `val` 을 넣는다.  전부 제자리 갱신.

        방금 리셋된 판(`_fresh`)은 **세 칸을 모두 현재 미분값으로** 채운다.
        그러지 않으면 이전 에피소드의 미분값으로 처음 두 스텝을 굴린다.
        """
        h = self._hist[key]
        h[2].copy_(h[1])                 # 뒤에서부터 밀어야 덮어쓰지 않는다
        h[1].copy_(h[0])
        h[0].copy_(val)
        fresh = self._fresh
        torch.where(fresh, val, h[1], out=h[1])
        torch.where(fresh, val, h[2], out=h[2])
        return h


def _moment_minus_gyro(mp: MassProps, w: Tensor, moment: Tensor) -> Tensor:
    p, q, r = w.unbind(-1)
    ixx, iyy, izz, ixz = mp.ixx, mp.iyy, mp.izz, mp.ixz
    # J w
    jw = torch.stack((ixx * p + ixz * r, iyy * q, ixz * p + izz * r), dim=-1)
    return moment - torch.cross(w, jw, dim=-1)


def _solve_inertia(mp: MassProps, m: Tensor) -> Tensor:
    """J^-1 m.  J = [[ixx,0,ixz],[0,iyy,0],[ixz,0,izz]] 의 해석적 역행렬."""
    ixx, iyy, izz, ixz = mp.ixx, mp.iyy, mp.izz, mp.ixz
    det = ixx * izz - ixz * ixz
    mx, my, mz = m.unbind(-1)
    return torch.stack(
        ((izz * mx - ixz * mz) / det, my / iyy, (ixx * mz - ixz * mx) / det), dim=-1)
