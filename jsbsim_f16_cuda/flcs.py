# SPDX-License-Identifier: GPL-3.0-or-later
"""f16.xml 의 `<flight_control>` 을 torch 배치로.

`fcs/*-cmd-norm` 네 개를 받아 공력이 먹는 조종면 여섯 개를 낸다.  JSBSim 안에서
120 Hz 로 도는 계층이고, 조종간 -> **여기** -> 조종면 순이다.

    aileron/elevator/rudder/throttle-cmd-norm  +  기체 상태
        -> fcs/aileron-pos-rad, fcs/elevator-pos-rad, fcs/rudder-pos-rad,
           fcs/lef-pos-rad, fcs/flaperon-mix-rad, fcs/speedbrake-pos-rad,
           fcs/throttle-pos-norm, gear/gear-pos-norm

원본은 JSBSim 패키지의 `aircraft/f16/f16.xml` 308~983 행.  아래 값은 전부
JSBSim 1.3.0 실측(200 채널)과 대조해서 확정한 것이고, `fdm_verify.py` 가 한 프레임
단위로 계속 확인한다 (조종면 여섯 개 오차 0).

--------------------------------------------------------------------------
프레임 안 실행 순서 -- 여기가 제일 조용히 틀리는 곳
--------------------------------------------------------------------------
JSBSim 한 프레임은

    propagate -> input -> inertial -> atmosphere -> winds -> **FCS**
    -> auxiliary -> propulsion -> aerodynamics -> ... -> accelerations

순서다.  FCS 는 auxiliary 와 accelerations 보다 **먼저** 돈다.  그래서

    한 프레임 늦게 읽는 것 (auxiliary/accelerations 산출물):
        aero/alpha-rad, velocities/{p,q,r}-aero-rad_sec, velocities/mach,
        velocities/vc-kts, velocities/vg-fps,
        accelerations/n-pilot-{y,z}-norm
    같은 프레임에서 읽는 것 (propagate 산출물):
        attitude/pitch-rad, attitude/roll-rad

실측으로 확인했다 (random case0, 7200 프레임):
    alpha-limiter-norm vs 1.0472*alpha  --  같은 프레임 4.5e-3 / 이전 프레임 3.0e-8
    n-pilot-z-correction vs cos(th)cos(phi) -- 같은 프레임 1.2e-7 / 이전 2.2e-3
한 스텝 지연을 빼먹으면 g 루프 위상이 어긋나 고하중에서 벌어진다.  그래서 이
클래스가 지연 버퍼를 **스스로** 들고 있다 -- 호출자는 언제나 *이번 프레임* 값을
넘기면 되고, 어디서 한 프레임 미룰지는 여기서 정한다.

--------------------------------------------------------------------------
JSBSim `<pid>` 의 실제 동작 (f16 에서는 적분기가 죽어 있다)
--------------------------------------------------------------------------
FGPID::Run 은 trigger 가 **0 일 때만** 적분한다 (0 이 아니면 정지, 음수면 리셋).
f16.xml 세 PID 의 trigger 는 전부 "기본 1, 저속일 때 0" 이라 **정상 비행에서는
적분이 통째로 멈춘다**.  즉 ki(5e-4 / 0.025 / 1e-5)는 공중전 구간에서 아무 일도
하지 않는다.  실측:

    roll-rate-pid  vs  kp*e + kd*(e-e_prev)/dt      max 2.6e-7   (ki 포함하면 2.3e-3)
    yaw-load-pid   vs  kp*e + kd*(e-e_prev)/dt      max 8.2e-8
    g-load-pid     vs  clip(kp*e, +-1)              max 6.3e-6

g-load-pid 의 6.25e-6 은 리셋 직후 vc 가 5 kt 미만이던 한두 프레임에 쌓인
적분 잔재가 그대로 얼어붙은 것이다.  `i_pitch0` 로 넣을 수 있게 열어 뒀지만
기본값은 0 이다 (조종면으로 2.7e-6 rad).

미분항은 3 점 후진차분이 아니라 **단순 후진차분** `(e - e_prev)/dt` 다.
적분 방식은 ki 에 type 속성이 없으면 Adams-Bashforth 2 차가 기본이다 -- 여기서는
안 돌지만 저속 진입 대비로 그대로 옮겨 놨다.

--------------------------------------------------------------------------
러더의 함정 -- 두 컴포넌트가 같은 프로퍼티에 쓴다
--------------------------------------------------------------------------
`fcs/yaw-load-pid` 의 <output> 과 `fcs/rudder-position` kinematic 의 <output> 이
**둘 다 `fcs/rudder-pos-norm`** 이다.  FGKinematic 은 Run() 첫머리에 자기 출력
프로퍼티를 다시 읽으므로, 러더 레이트 제한기의 출발점은 "직전 프레임의 러더
위치" 가 아니라 **이번 프레임의 PID 출력**이다.  실측 (random case0):

    출발 = 직전 rudder-pos-norm   max 1.25e-1   <- 틀림
    출발 = 이번 프레임 yaw-load-pid  max 7.1e-8   <- 맞음

덕분에 러더 kinematic 은 **상태가 없다**.  그리고 무입력에서 러더가 12 deg 쯤에
앉아 있는 것이 여기서 나온다.  JSBSim 번들 f16 의 성질이라 재현 대상이지 고칠 대상이 아니다.

--------------------------------------------------------------------------
공력 계약면
--------------------------------------------------------------------------
f16.xml 의 <aerodynamics> 가 실제로 읽는 FCS 출력은 여섯 개뿐이다:
`aileron-pos-rad`, `elevator-pos-rad`, `rudder-pos-rad`, `lef-pos-rad`,
`flaperon-mix-rad`, `speedbrake-pos-rad`.  추진은 `throttle-pos-norm`,
착륙장치 항력은 `gear/gear-pos-norm`.

두 가지가 놀랍다.
  - `aileron-pos-rad` 는 kinematic 을 **거치지 않는다**.  `roll-rate-command`
    (PID 출력) 를 바로 0.375 rad 로 스케일한 값이다.  0.3 s 레이트 제한이 걸린
    `aileron-position` 은 플래퍼론 믹스와 좌우 에일러론 표시값으로만 간다.
  - `speedbrake-pos-rad` 에 쓰는 컴포넌트가 하나도 없다.  Speedbrake 채널은
    deg/norm 만 낸다.  그래서 **항상 0** 이다 (실측 확인).
그리고 JSBSim 번들 f16 은 로드 직후 **착륙장치가 내려가 있고**(`gear/gear-pos-norm`
= 1.0) 이 포팅도 그 상태로 고정이다.  lef 스위치의 `gear == 0 -> 0.436` 분기가
그래서 죽고, 받음각 15 deg 를 넘겨도 앞전 플랩은 0.262 rad 에서 멈춘다.  공력의
착륙장치 항력도 상시 켜져 있다 (`aero.AeroState.gear`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor

# --- 상수 (전부 f16.xml 에서 그대로) -----------------------------------------
DT_DEFAULT = 1.0 / 120.0

ROLL_RATE_GAIN = 0.31821          # fcs/roll-rate-norm
ROLL_KP, ROLL_KI, ROLL_KD = 3.0, 0.00050, -0.00125
ROLL_TRIGGER_VC = 20.0            # vc < 20 kt 이면 trigger 0 = 적분 켜짐

AILERON_RANGE_RAD = 0.375         # aerosurface_scale
AILERON_RATE = 2.0 / 0.3          # kinematic, norm/s
FLAPERON_MIX_GAIN = 1.4324

PITCH_KP, PITCH_KI, PITCH_KD = 0.3000, 0.0250, 0.0
PITCH_TRIGGER_VC = 5.0
# g-load-pid 적분기에 얼어붙어 있는 잔재.  `JSBSimF16.reset` 의 순서
# (set_controls(0) -> run_ic -> reset_to_initial_conditions -> do_trim) 를 도는
# 동안 vc 가 5 kt 미만인 프레임이 한두 개 끼어 그때만 적분이 돌고, 그 뒤로는
# trigger 가 1 이라 영영 멈춘다.  다섯 데이터셋 전부에서 값이 같다(6.25e-6).
# 조종면으로는 2.7e-6 rad = 1.6e-4 deg 라 물리적으로는 없는 것과 같지만,
# 기준 궤적과 자릿수까지 맞추려면 넣어야 한다.  torch 쪽에서 트림 절차를
# 재현하지 않을 것이므로 상수로 박는다.
I_PITCH_RESIDUAL = 6.25e-6
PITCH_RATE_GAIN = 6.2             # fcs/pitch-rate-norm
G_LOAD_GAIN = 0.020               # fcs/g-load-norm
ALPHA_LIMITER_GAIN = 1.0472       # fcs/alpha-limiter-norm
ELEV_CMD_MIN, ELEV_CMD_MAX = -1.0, 0.44   # 9 g / 4 g 비대칭
ELEVATOR_RANGE_RAD = 0.436
ELEVATOR_RATE = 2.0 / 0.3
DHT_LIMIT_RAD = 0.436

YAW_KP, YAW_KI, YAW_KD = 0.105500, 0.000010, 0.00005
YAW_TRIGGER_VC = 10.0
YAW_LOAD_GAIN = 0.25
RUDDER_RANGE_RAD = 0.524
RUDDER_RATE = 2.0 / 0.4

TEF_NORM_GAIN = 2.864789
TEF_RATE = 1.0 / 3.0              # kinematic detent [0,1], time 3 s
TEF_VC_KT = 250.0                 # vc < 250 -> 0.349 rad
TEF_HI_MACH_RAD = -0.0349

LEF_ALPHA_LO = 0.0873             # rad, 5.0 deg  -> 0.262
LEF_ALPHA_HI = 0.2618             # rad, 15.0 deg -> 0.436 (착륙장치 올렸을 때만)
LEF_HI_MACH_RAD = -0.0349
MACH_HI = 0.9

# 받음각 스케줄 표 (fcs/elevator-scheduler)
#   -0.5236 -> 0.0 | -0.5 -> 0.11 | 0.0 -> 1.0 | 0.5 -> 0.11 | 0.5236 -> 0.0
SCHED_A1 = 0.5
SCHED_A2 = 0.5236
SCHED_SLOPE1 = (1.0 - 0.11) / SCHED_A1          # 1.78
SCHED_Y1 = 0.11


def _pitch_authority(alpha_abs: Tensor) -> Tensor:
    """fcs/elevator-scheduler 의 게인.  |alpha| 에 대칭인 구간선형 표."""
    a = alpha_abs.clamp(max=SCHED_A2)
    inner = 1.0 - SCHED_SLOPE1 * a
    outer = SCHED_Y1 * (SCHED_A2 - a) * (1.0 / (SCHED_A2 - SCHED_A1))
    return torch.where(a <= SCHED_A1, inner, outer)


def _mach_gain(mach: Tensor) -> Tensor:
    """fcs/aileron-speed-compensated 의 게인.  mach 0 -> 1.0, 1.0 -> 0.15, 물림."""
    return 1.0 - 0.85 * mach.clamp(0.0, 1.0)


def _vg_gain(vg_fps: Tensor) -> Tensor:
    """fcs/yaw-rate-norm 의 게인.  80 -> 0, 100 -> 15, 150 -> 100, 물림.

    공중전 구간(vg > 150 fps = 89 kt)에서는 언제나 100 이다.  그래도 옮겨 두는
    이유는 이 게인이 100 까지 튀기 때문에 -- 저속에서 한 번이라도 물리면 요
    채널 전체가 달라진다.
    """
    seg1 = ((vg_fps - 80.0) * (1.0 / 20.0)).clamp(0.0, 1.0) * 15.0
    seg2 = ((vg_fps - 100.0) * (1.0 / 50.0)).clamp(0.0, 1.0) * 85.0
    return seg1 + seg2


def _ratelim(pos: Tensor, target: Tensor, max_delta: float) -> Tensor:
    """FGKinematic 한 프레임.  detent 두 개짜리(=구간 하나)일 때의 형태."""
    return pos + (target - pos).clamp(-max_delta, max_delta)


def _ratelim_(pos: Tensor, target: Tensor, max_delta: float) -> None:
    """같은 것, 제자리에서.  상태 텐서의 주소를 유지해야 그래프로 캡처된다."""
    pos.add_((target - pos).clamp_(-max_delta, max_delta))


@dataclass
class FlcsOutput:
    """공력·추진이 읽는 것 + 검증용 내부 신호."""
    aileron_pos_rad: Tensor
    elevator_pos_rad: Tensor
    rudder_pos_rad: Tensor
    lef_pos_rad: Tensor
    flaperon_mix_rad: Tensor
    speedbrake_pos_rad: Tensor
    throttle_pos_norm: Tensor
    gear_pos_norm: Tensor
    # --- 아래는 진단용 (aero 는 안 읽는다) ---
    roll_rate_command: Tensor | None = None
    roll_rate_pid: Tensor | None = None
    pitch_scheduler: Tensor | None = None
    elevator_scheduler: Tensor | None = None
    alpha_limiter_norm: Tensor | None = None
    g_load_pid: Tensor | None = None
    yaw_load_pid: Tensor | None = None
    yaw_scheduler: Tensor | None = None
    aileron_position: Tensor | None = None
    aileron_speed_compensated: Tensor | None = None
    tef_control: Tensor | None = None
    dht_left_pos_rad: Tensor | None = None
    dht_right_pos_rad: Tensor | None = None


class BatchFLCS:
    """배치 B 대의 F-16 FLCS.  상태는 전부 (B,) 텐서.

    한 프레임:

        out = flcs.step(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd,
                        alpha_rad=..., p_aero=..., q_aero=..., r_aero=...,
                        n_pilot_z_norm=..., n_pilot_y_norm=...,
                        mach=..., vc_kts=..., vg_fps=...,
                        theta_rad=..., phi_rad=...)

    **명령 부호는 JSBSim 규약 그대로다** -- `fcs/elevator-cmd-norm` 은 음수가
    기수 올림이고 `fcs/rudder-cmd-norm` 은 양수가 좌선회다.  조종간 규약
    ("+ = 당김 / 오른쪽") 에서 오려면 `from_stick()` 을 써라.  `n_pilot_z_norm`
    도 JSBSim 부호 그대로 -- **수평비행에서 -1** 이다 (+1 g 로 뒤집은 하중배수를
    그대로 넣으면 안 된다).

    상태를 갖는 요소:
        kinematic  aileron-position, elevator-position-normalized, tef-control
                   (rudder-position 은 상태가 없다 -- 위 설명 참조)
        pid        roll / pitch / yaw 각각 (적분 누적, 직전 오차)
        센서 지연  auxiliary/accelerations 산출물 9 개의 한 프레임 전 값
    """

    N_STATE_KEYS = (
        "ail_pos_norm", "ele_pos_norm", "tef_control",
        "i_roll", "e_roll_prev", "i_pitch", "e_pitch_prev",
        "i_yaw", "e_yaw_prev",
        "d_alpha", "d_p", "d_q", "d_r", "d_nz", "d_ny",
        "d_mach", "d_vc", "d_vg",
    )

    def __init__(self, batch: int, device="cpu", dtype=torch.float32,
                 dt: float = DT_DEFAULT, gear_pos_norm: float = 1.0,
                 gear_wow: float = 0.0, fbw_override: bool = False,
                 i_pitch0: float = I_PITCH_RESIDUAL,
                 fast_pid: bool = True) -> None:
        """`fast_pid=True` (기본) 면 PID 적분항 세 개를 아예 계산하지 않는다.

        f16 의 trigger 는 `vc < {20, 5, 10} kt` 일 때만 0 이고, 적분은 그때만
        돈다.  공중전 포락선은 300~650 kt 이라 **절대 안 걸린다** (기준 궤적
        205,000 프레임에서 최저 vc 가 184 kt).  끄면 프레임당 커널이 12 개
        줄어든다.  지상 활주까지 시뮬레이션할 일이 생기면 꺼라.
        """
        self.B = int(batch)
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)
        self.fbw_override = bool(fbw_override)
        self.fast_pid = bool(fast_pid)
        self.i_pitch0 = float(i_pitch0)
        # 착륙장치는 상수다 (내려간 채 -- 모듈 문서 참조).  텐서로 들 이유가 없다.
        self.gear_pos = float(gear_pos_norm)
        self.gear_wow = float(gear_wow)
        z = torch.zeros(self.B, device=self.device, dtype=self.dtype)
        # 상태 18 개를 **한 덩어리 (18, B) 로 잡고 행을 뷰로 나눠 준다.**  이름으로
        # 쓰는 것은 그대로지만, 마스크 리셋이 커널 18 개가 아니라 하나로 끝난다
        # (그래프 안에서는 리셋을 건너뛸 수가 없다 -- `mask.any()` 가 호스트
        # 동기화라 캡처가 깨지므로 매 프레임 무조건 돈다).
        self._state = torch.zeros(len(self.N_STATE_KEYS), self.B,
                                  device=self.device, dtype=self.dtype)
        for i, k in enumerate(self.N_STATE_KEYS):
            setattr(self, k, self._state[i])
        self._gear_t = torch.full((self.B,), self.gear_pos,
                                  device=self.device, dtype=self.dtype)
        self._zero = z.clone()
        # 스위치가 고르는 상수들.  매 프레임 `full_like` 로 만들면 그만큼
        # 커널이 더 뜬다 -- 제어 계층은 커널 하나가 4,096 원소짜리라 계산보다
        # 띄우는 값이 비싸다.
        def _k(v):
            return torch.full((self.B,), float(v), device=self.device,
                              dtype=self.dtype)
        self._c_tef = _k(0.349)
        self._c_hi_mach = _k(TEF_HI_MACH_RAD)
        self._c_lef_lo = _k(0.262)
        self._c_lef_hi = _k(0.436)
        self._ail_rate = AILERON_RATE * self.dt
        self._ele_rate = ELEVATOR_RATE * self.dt
        self._rud_rate = RUDDER_RATE * self.dt
        self._tef_rate = TEF_RATE * self.dt
        # 고정 출력 버퍼.  매 프레임 새로 할당하면 CUDA 그래프 재생 때 주소가
        # 어긋나고, 호출자가 들고 있던 텐서가 낡은 값을 가리키게 된다.
        self._out = FlcsOutput(
            aileron_pos_rad=z.clone(), elevator_pos_rad=z.clone(),
            rudder_pos_rad=z.clone(), lef_pos_rad=z.clone(),
            flaperon_mix_rad=z.clone(), speedbrake_pos_rad=z.clone(),
            throttle_pos_norm=z.clone(), gear_pos_norm=self._gear_t)
        self.reset()

    # -- 리셋 ----------------------------------------------------------------

    def reset(self, idx: Tensor | None = None, mask: Tensor | None = None, *,
              alpha_rad=None, p_aero=None,
              q_aero=None, r_aero=None, n_pilot_z_norm=None,
              n_pilot_y_norm=None, mach=None, vc_kts=None, vg_fps=None,
              ail_pos_norm=None, ele_pos_norm=None, tef_control=None) -> None:
        """FCS 내부 상태를 민다.

        JSBSim 도 같은 함정이 있다 -- `run_ic()` 는 FCS 적분기를 안 지운다.  안
        지우면 직전 비행에서 세게 당긴 기체가 다음 비행 첫 프레임에 그 상태를 물고
        시작한다.

            mask=(B,) bool   **그래프 안에서 쓸 수 있는 형태.**  `masked_fill_`
                             과 `torch.where` 만 쓰고 호스트 동기화를 안 한다.
                             기체마다 제각각 리셋되는 배치 루프는 이쪽을 써라.
            idx=(n,) long    호스트 쪽 편의용.  `nonzero()` 로 만든 인덱스는
                             동기화를 부르므로 그래프에 못 넣는다.
            둘 다 없음        전부 리셋.

        둘 다 주면 `mask` 가 이긴다.  씨앗 키워드는 (B,) 텐서로 주면 되고
        마스크가 켜진 판에만 들어간다.

        지연 버퍼는 기본이 0 이다.  JSBSim 은 트림 직후라 0 이 아니므로, 첫
        프레임까지 정확히 맞추려면 트림 상태를 키워드로 넣어 씨앗을 준다.
        (한 프레임이면 대부분 씻겨 나가지만, 트림값과 대조할 때는 차이가 보인다.)
        """
        seeds = dict(d_alpha=alpha_rad, d_p=p_aero, d_q=q_aero, d_r=r_aero,
                     d_nz=n_pilot_z_norm, d_ny=n_pilot_y_norm, d_mach=mach,
                     d_vc=vc_kts, d_vg=vg_fps, ail_pos_norm=ail_pos_norm,
                     ele_pos_norm=ele_pos_norm, tef_control=tef_control)
        if not any(v is not None for v in seeds.values()):
            # 씨앗 없는 보통의 리셋 -- 뭉친 버퍼째 한 번에 민다 (커널 1~2 개).
            # 배치 루프가 매 프레임 부르는 경로라 여기가 빨라야 한다.
            if mask is not None:
                self._state.masked_fill_(mask, 0.0)
                if self.i_pitch0 != 0.0:
                    self.i_pitch.masked_fill_(mask, self.i_pitch0)
            elif idx is None:
                self._state.zero_()
                if self.i_pitch0 != 0.0:
                    self.i_pitch.fill_(self.i_pitch0)
            else:
                self._state[:, idx] = 0.0
                if self.i_pitch0 != 0.0:
                    self.i_pitch[idx] = self.i_pitch0
            return
        for k in self.N_STATE_KEYS:
            v = getattr(self, k)
            fill = seeds.get(k)
            if fill is None:
                fill = self.i_pitch0 if k == "i_pitch" else 0.0
            if mask is not None:
                if torch.is_tensor(fill):
                    v.copy_(torch.where(mask, fill.to(v.dtype), v))
                else:
                    v.masked_fill_(mask, fill)
            elif idx is None:
                if torch.is_tensor(fill):
                    v.copy_(fill.to(v))
                else:
                    v.fill_(fill)
            else:
                if torch.is_tensor(fill):
                    v[idx] = fill.to(v)
                else:
                    v[idx] = fill

    def state_dict(self) -> dict[str, Tensor]:
        return {k: getattr(self, k).clone() for k in self.N_STATE_KEYS}

    def load_state_dict(self, d: dict[str, Tensor]) -> None:
        for k, v in d.items():
            getattr(self, k).copy_(v)

    # -- 부호 규약 변환 -------------------------------------------------------

    @staticmethod
    def from_stick(aileron: Tensor, elevator: Tensor, rudder: Tensor,
                   throttle: Tensor):
        """조종간 규약("+ = 당김 / 오른쪽") -> JSBSim `fcs/*-cmd-norm`.

        `f16_reference.JSBSimF16Ref.set_controls` 와 **같은 식**이다: 에일러론은 ±1 로
        자르고, 엘리베이터와 러더는 자른 뒤 부호를 뒤집고, 스로틀은 [0,1] 로 자른다.
        """
        return (aileron.clamp(-1.0, 1.0),
                -elevator.clamp(-1.0, 1.0),
                -rudder.clamp(-1.0, 1.0),
                throttle.clamp(0.0, 1.0))

    # -- 한 프레임 -----------------------------------------------------------

    def step(self, aileron_cmd: Tensor, elevator_cmd: Tensor,
             rudder_cmd: Tensor, throttle_cmd: Tensor, *,
             alpha_rad: Tensor, p_aero: Tensor, q_aero: Tensor, r_aero: Tensor,
             n_pilot_z_norm: Tensor, n_pilot_y_norm: Tensor,
             mach: Tensor, vc_kts: Tensor, vg_fps: Tensor,
             theta_rad: Tensor, phi_rad: Tensor,
             pitch_trim_cmd: Tensor | float = 0.0,
             roll_trim_cmd: Tensor | float = 0.0,
             yaw_trim_cmd: Tensor | float = 0.0,
             diagnostics: bool = False) -> FlcsOutput:
        """f16.xml 채널을 정의 순서대로 한 번 돌린다.

        auxiliary/accelerations 산출물은 **이번 프레임 값을 그대로 넘겨라** --
        한 프레임 미루는 일은 여기서 한다.  attitude 두 개만 이번 프레임 값이
        곧바로 쓰인다.
        """
        dt = self.dt
        # 이번 프레임 FCS 가 실제로 보는 값 = 한 프레임 전 것
        a_d, p_d, q_d, r_d = self.d_alpha, self.d_p, self.d_q, self.d_r
        nz_d, ny_d = self.d_nz, self.d_ny
        mach_d, vc_d, vg_d = self.d_mach, self.d_vc, self.d_vg

        # ============================ 1. Flaps ============================
        tef_pos_rad = torch.where(
            vc_d < TEF_VC_KT, self._c_tef,
            torch.where(mach_d > MACH_HI, self._c_hi_mach, self._zero))
        tef_target = (tef_pos_rad * TEF_NORM_GAIN).clamp(-1.0, 1.0)
        # detent [-1, 0, 1] / time [3, 0, 3]: [0,1] 구간만 1/3 /s 로 제한되고
        # [-1,0] 구간은 시간 0 = 즉시다.  둘 다 지나는 경우 0 에서 멈추지 않고
        # 남은 dt 로 계속 간다.
        self._tef_kinematic(self.tef_control, tef_target)

        # ============================ 2. Roll =============================
        roll_rate_norm = ROLL_RATE_GAIN * p_d
        e_roll = aileron_cmd - roll_rate_norm
        roll_pid = self._pid(e_roll, "roll", ROLL_KP, ROLL_KI, ROLL_KD,
                             vc_d, ROLL_TRIGGER_VC, clip=False)
        roll_rate_command = (roll_pid + aileron_cmd).clamp_(-1.0, 1.0)
        aileron_pos_rad = torch.mul(roll_rate_command, AILERON_RANGE_RAD,
                                    out=self._out.aileron_pos_rad)
        rrc_switch = aileron_cmd if self.fbw_override else roll_rate_command
        _ratelim_(self.ail_pos_norm, rrc_switch, self._ail_rate)
        ail_sc = self.ail_pos_norm * _mach_gain(mach_d)
        left_flaperon = (-self.tef_control - ail_sc).clamp_(-1.0, 1.0)
        right_flaperon = (self.tef_control - ail_sc).clamp_(-1.0, 1.0)
        flaperon_mix_rad = torch.mul(left_flaperon.add_(right_flaperon),
                                     FLAPERON_MIX_GAIN,
                                     out=self._out.flaperon_mix_rad)
        left_ail_rad = ail_sc * AILERON_RANGE_RAD
        right_ail_rad = -left_ail_rad

        # ============================ 3. Pitch ============================
        nz_corr = torch.cos(theta_rad) * torch.cos(phi_rad)
        g_load_corrected = nz_d - nz_corr
        elev_lim = (elevator_cmd + pitch_trim_cmd).clamp(ELEV_CMD_MIN,
                                                         ELEV_CMD_MAX)
        elev_sched = elev_lim * _pitch_authority(a_d.abs())
        alpha_limiter_norm = ALPHA_LIMITER_GAIN * a_d
        e_pitch = (elev_sched + PITCH_RATE_GAIN * q_d
                   - G_LOAD_GAIN * g_load_corrected)
        g_load_pid = self._pid(e_pitch, "pitch", PITCH_KP, PITCH_KI, PITCH_KD,
                               vc_d, PITCH_TRIGGER_VC, clip=True)
        pitch_scheduler = (elev_sched + alpha_limiter_norm
                           + g_load_pid).clamp(-1.0, 1.0)
        ps_switch = elev_lim if self.fbw_override else pitch_scheduler
        _ratelim_(self.ele_pos_norm, ps_switch, self._ele_rate)
        elevator_pos_rad = torch.mul(self.ele_pos_norm, ELEVATOR_RANGE_RAD,
                                     out=self._out.elevator_pos_rad)

        # ============================ 4. Yaw ==============================
        yaw_rate_norm = r_d * _vg_gain(vg_d)
        e_yaw = rudder_cmd + yaw_rate_norm + YAW_LOAD_GAIN * ny_d
        yaw_load_pid = self._pid(e_yaw, "yaw", YAW_KP, YAW_KI, YAW_KD,
                                 vc_d, YAW_TRIGGER_VC, clip=True)
        yaw_scheduler = (rudder_cmd + yaw_trim_cmd
                         + yaw_load_pid).clamp_(-1.0, 1.0)
        # kinematic 의 출발점은 방금 PID 가 같은 프로퍼티에 쓴 값이다 (상태 없음)
        rudder_pos_norm = _ratelim(yaw_load_pid, yaw_scheduler, self._rud_rate)
        rudder_pos_rad = torch.mul(rudder_pos_norm, RUDDER_RANGE_RAD,
                                   out=self._out.rudder_pos_rad)

        # ======================= 6. Leading Edge Flap =====================
        lef_pos_rad = self._out.lef_pos_rad
        lef_pos_rad.copy_(self._lef(a_d, mach_d))

        # ============================ 7. Throttle =========================
        throttle_pos_norm = torch.mul(throttle_cmd, 2.0,
                                      out=self._out.throttle_pos_norm)

        # 지연 버퍼 갱신 -- 다음 프레임이 이번 프레임 값을 본다.
        # **반드시 복사다.**  참조만 잡으면, 호출자가 입력 버퍼를 고정해 두고
        # 매 프레임 그 자리에 덮어쓰는 순간(정적 버퍼 / CUDA 그래프 방식이
        # 정확히 그렇다) 지연 버퍼가 이번 프레임 값을 가리키게 되어 한 프레임
        # 지연이 조용히 사라진다.  그러면 g 루프 위상이 어긋나는데, 눈에 띄는
        # 것은 고하중 기동에서뿐이라 찾기가 아주 나쁘다.
        self.d_alpha.copy_(alpha_rad)
        self.d_p.copy_(p_aero)
        self.d_q.copy_(q_aero)
        self.d_r.copy_(r_aero)
        self.d_nz.copy_(n_pilot_z_norm)
        self.d_ny.copy_(n_pilot_y_norm)
        self.d_mach.copy_(mach)
        self.d_vc.copy_(vc_kts)
        self.d_vg.copy_(vg_fps)

        # 출력은 고정 버퍼에 **직접** 썼다 (`out=`) -- CUDA 그래프로 캡처하려면
        # 주소가 변하면 안 되고, 따로 복사하면 커널이 그만큼 더 뜬다.
        out = self._out
        if diagnostics:
            out.roll_rate_command = roll_rate_command
            out.roll_rate_pid = roll_pid
            out.pitch_scheduler = pitch_scheduler
            out.elevator_scheduler = elev_sched
            out.alpha_limiter_norm = alpha_limiter_norm
            out.g_load_pid = g_load_pid
            out.yaw_load_pid = yaw_load_pid
            out.yaw_scheduler = yaw_scheduler
            out.aileron_position = self.ail_pos_norm
            out.aileron_speed_compensated = ail_sc
            out.tef_control = self.tef_control
            out.dht_left_pos_rad = (-elevator_pos_rad - left_ail_rad).clamp(
                -DHT_LIMIT_RAD, DHT_LIMIT_RAD)
            out.dht_right_pos_rad = (elevator_pos_rad + right_ail_rad).clamp(
                -DHT_LIMIT_RAD, DHT_LIMIT_RAD)
        return out

    # -- 요소 ----------------------------------------------------------------

    def _pid(self, e: Tensor, name: str, kp: float, ki: float, kd: float,
             vc: Tensor, trigger_vc: float, clip: bool) -> Tensor:
        """FGPID::Run 을 그대로.

        trigger != 0 이면 적분 정지, < 0 이면 리셋.  f16 의 trigger 는
        `vc < trigger_vc` 일 때만 0 이므로 **정상 비행에서는 적분이 멈춘다**.
        적분 방식은 Adams-Bashforth 2 차(ki 에 type 속성이 없을 때의 기본값),
        미분은 단순 후진차분.
        """
        i_acc = getattr(self, f"i_{name}")
        e_prev = getattr(self, f"e_{name}_prev")
        dval = (e - e_prev) * (1.0 / self.dt)
        if ki != 0.0 and not self.fast_pid:
            # trigger == 0 인 곳만.  공중전 구간에서는 한 번도 안 걸린다.
            delta = torch.where(vc < trigger_vc, 1.5 * e - 0.5 * e_prev,
                                torch.zeros_like(e))
            i_acc.add_(delta, alpha=ki * self.dt)
        e_prev.copy_(e)                   # 주소 유지 (그래프 캡처 때문에)
        out = kp * e + i_acc + kd * dval
        return out.clamp_(-1.0, 1.0) if clip else out

    def _tef_kinematic(self, pos: Tensor, target: Tensor) -> None:
        """detent [-1, 0, 1], time [3, 0, 3] 짜리 FGKinematic.

        [0, 1] 구간만 1/3 norm/s 로 제한되고 [-1, 0] 구간은 시간이 0 이라
        즉시 지나간다.  `mach > 0.9` 에서 뒷전 플랩이 -0.0349 rad 로 튀는
        분기가 이 음수 구간을 쓴다 -- 650 kt / 20,000 ft 는 mach 1.11 이므로
        공중전에서 실제로 들어간다.
        """
        pos_pos = pos.clamp_min(0.0)               # 양수 구간에 남은 몫
        tgt_pos = target.clamp_min(0.0)
        moved = pos_pos + (tgt_pos - pos_pos).clamp_(-self._tef_rate,
                                                     self._tef_rate)
        # 목표가 음수면 양수 구간을 레이트로 내려온 뒤 0 아래는 즉시
        pos.copy_(torch.where((target < 0.0) & (moved <= 0.0), target, moved))

    def _lef(self, alpha: Tensor, mach: Tensor) -> Tensor:
        """fcs/lef-pos-rad 스위치.  위에서부터 첫 참인 test.

        기본 0.0
          1) WOW==1 AND gear>0            -> -0.0349
          2) gear==0 AND alpha>0.2618     ->  0.436
          3) WOW==0 AND alpha>0.0873      ->  0.262
          4) mach>0.9                     -> -0.0349

        레이트 제한이 **없다** -- 받음각 5 deg 를 넘는 순간 계단으로 튄다.
        착륙장치가 내려가 있어(모듈 문서) 2) 가 죽고 15 deg 위로도 0.262 다.
        """
        if self.gear_wow == 1.0 and self.gear_pos > 0.0:
            return self._c_hi_mach            # test 1 이 항상 이긴다
        out = torch.where(mach > MACH_HI, self._c_hi_mach, self._zero)
        if self.gear_wow == 0.0:
            out = torch.where(alpha > LEF_ALPHA_LO, self._c_lef_lo, out)
        if self.gear_pos == 0.0:
            out = torch.where(alpha > LEF_ALPHA_HI, self._c_lef_hi, out)
        return out
