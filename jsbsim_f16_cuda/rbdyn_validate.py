# SPDX-License-Identifier: GPL-3.0-or-later
"""`rbdyn.py` 를 JSBSim 과 나란히 돌려 궤적 발산을 잰다.

**힘 주입 방식.**  매 스텝 JSBSim 에서 body 축 힘·모멘트·질량·관성을 읽어 우리
적분기에 그대로 먹인다.  그러면 남는 차이는 **적분 방식과 운동방정식 뿐**이다 --
공력 모델 오차가 섞이지 않는다.  공력·추력 모듈이 완성되면 같은 틀에서 주입만
빼면 전체 검증이 된다.

실행::

    python jsbsim_f16_cuda/rbdyn_validate.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch


from jsbsim_f16_cuda.f16_reference import JSBSimF16Ref  # noqa: E402


def JSBSimF16():
    """`rbdyn` 기본 위도(`CENTER_LAT_DEG`)에 맞춘 JSBSim 표준 기준기."""
    from jsbsim_f16_cuda.rbdyn import CENTER_LAT_DEG
    return JSBSimF16Ref(lat0_deg=CENTER_LAT_DEG, lon0_deg=126.978)

from jsbsim_f16_cuda.rbdyn import (  # noqa: E402
    FT, G0, LBF, LBFFT, SLUG, SLUGFT2, MassProps, RigidBody6DOF, Wrench,
    dcm_l2b, euler_from_quat, quat_from_euler,
)

DT = 1.0 / 120.0
DEG = 180.0 / math.pi
DTYPE = torch.float64        # 검증은 배정밀도.  적분 오차와 반올림 오차를 섞지 않는다.


# ---------------------------------------------------------------------------
def read_jsbsim(fdm) -> dict:
    """JSBSim 한 프레임 전체를 SI/NED 로 뽑는다."""
    lat = math.radians(fdm["position/lat-geod-deg"])
    lon = fdm["position/long-gc-rad"]
    return dict(
        # 위치는 기준기(`f16_reference.JSBSimF16Ref.xy`)와 같은 국소 접평면 투영을 쓴다
        lat=lat, lon=lon,
        h=fdm["position/h-sl-ft"] * FT,
        uvw=np.array([fdm["velocities/u-fps"], fdm["velocities/v-fps"],
                      fdm["velocities/w-fps"]]) * FT,
        pqr=np.array([fdm["velocities/p-rad_sec"], fdm["velocities/q-rad_sec"],
                      fdm["velocities/r-rad_sec"]]),
        euler=np.array([fdm["attitude/phi-rad"], fdm["attitude/theta-rad"],
                        fdm["attitude/psi-rad"]]),
        force=np.array([fdm["forces/fbx-total-lbs"], fdm["forces/fby-total-lbs"],
                        fdm["forces/fbz-total-lbs"]]) * LBF,
        moment=np.array([fdm["moments/l-total-lbsft"], fdm["moments/m-total-lbsft"],
                         fdm["moments/n-total-lbsft"]]) * LBFFT,
        mass=fdm["inertia/mass-slugs"] * SLUG,
        ixx=fdm["inertia/ixx-slugs_ft2"] * SLUGFT2,
        iyy=fdm["inertia/iyy-slugs_ft2"] * SLUGFT2,
        izz=fdm["inertia/izz-slugs_ft2"] * SLUGFT2,
        ixz=fdm["inertia/ixz-slugs_ft2"] * SLUGFT2,
        vned=np.array([fdm["velocities/v-north-fps"], fdm["velocities/v-east-fps"],
                       fdm["velocities/v-down-fps"]]) * FT,
    )


R_EARTH = 6371000.0


def ned_pos(snap: dict, lat0: float, lon0: float) -> np.ndarray:
    n = (snap["lat"] - lat0) * R_EARTH
    e = (snap["lon"] - lon0) * R_EARTH * math.cos(lat0)
    return np.array([n, e, -snap["h"]])


def sync(body: RigidBody6DOF, snap: dict, lat0: float, lon0: float, idx=0) -> None:
    """우리 적분기 상태를 JSBSim 스냅샷에 정확히 맞춘다 (배치 원소 `idx`)."""
    t = lambda a: torch.as_tensor(a, dtype=body.dtype, device=body.device)
    body.pos_ned[idx] = t(ned_pos(snap, lat0, lon0))
    body.uvw[idx] = t(snap["uvw"])
    body.pqr[idx] = t(snap["pqr"])
    e = snap["euler"]
    body.quat[idx] = quat_from_euler(t(e[0]), t(e[1]), t(e[2]))


def att_err_deg(q_ours: torch.Tensor, euler_ref: np.ndarray) -> tuple[float, np.ndarray]:
    """자세 오차.  (전체 각거리, 축별 오차) [deg].

    전체 각거리는 쿼터니언 측지거리다 -- 오일러각은 수직 근처에서 특이해
    축별 차이만 보면 멀쩡한데 튄 것처럼 보인다 (수직 근처 오일러 특이점).
    """
    qr = quat_from_euler(*[torch.as_tensor(v, dtype=q_ours.dtype) for v in euler_ref])
    d = abs(float(torch.dot(q_ours.cpu().double(), qr.double())))
    total = 2.0 * math.degrees(math.acos(min(1.0, d)))
    phi, th, psi = euler_from_quat(q_ours.cpu())
    ours = np.array([float(phi), float(th), float(psi)])
    diff = (ours - euler_ref + math.pi) % (2 * math.pi) - math.pi
    return total, np.degrees(diff)


# ---------------------------------------------------------------------------
#: 60 s 를 버티면서도 세 축이 전부 살아 있는 고정 입력.  실측으로 골랐다
#: (`(0, .5, .15, 1.0)` = 연속 루프, 고도 14.6k~30.4k ft, 속도 303~689 kt).
#: 에일러론을 고정하면 JSBSim 의 F-16 FLCS 는 롤 **각속도** 명령이라 계속 굴러
#: 나선강하로 45 초 안에 전부 추락한다 -- 고정 입력 케이스를 짤 때의 함정이다.
FIXED_CTL = (0.0, 0.5, 0.15, 1.0)


def run_case(integrator: str, seconds: float, control_mode: str, seed: int = 0,
             earth_rotation: bool = True, frame_correction: bool = True,
             marks=(1.0, 10.0, 60.0)) -> dict:
    """JSBSim 과 우리 적분기를 나란히 돌린다.  주입 방식."""
    rng = np.random.default_rng(seed)
    be = JSBSimF16()
    be.reset(x=0.0, y=0.0, psi=0.4, v_kt=450.0)
    fdm = be._fdm
    lat0, lon0 = be._lat0, be._lon0

    ctl = FIXED_CTL
    be.set_controls(*ctl)
    # 트림 직후의 과도응답을 지나 보낸다.  두 적분기 모두 같은 지점에서 출발한다.
    for _ in range(120):
        fdm.run()

    body = RigidBody6DOF((1,), dt=DT, dtype=DTYPE, integrator=integrator,
                         earth_rotation=earth_rotation,
                         frame_correction=frame_correction)
    snap = read_jsbsim(fdm)
    sync(body, snap, lat0, lon0)

    n_steps = int(round(seconds / DT))
    mark_steps = {int(round(m / DT)): m for m in marks}
    out: dict[float, dict] = {}
    next_change = 0

    for k in range(n_steps):
        if control_mode == "random" and k >= next_change:
            ctl = (float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                   float(rng.uniform(-0.4, 0.4)), float(rng.uniform(0.3, 1.0)))
            be.set_controls(*ctl)
            next_change = k + int(round(float(rng.uniform(1.0, 3.0)) / DT))

        snap = read_jsbsim(fdm)
        mp = MassProps(
            mass=torch.full((1,), snap["mass"], dtype=DTYPE),
            ixx=torch.full((1,), snap["ixx"], dtype=DTYPE),
            iyy=torch.full((1,), snap["iyy"], dtype=DTYPE),
            izz=torch.full((1,), snap["izz"], dtype=DTYPE),
            ixz=torch.full((1,), snap["ixz"], dtype=DTYPE),
        )
        w = Wrench(
            force=torch.as_tensor(snap["force"], dtype=DTYPE).unsqueeze(0),
            moment=torch.as_tensor(snap["moment"], dtype=DTYPE).unsqueeze(0),
        )
        body.step(w, mp)
        fdm.run()

        if not (0.0 < fdm["position/h-sl-ft"] < 100000.0):
            out["crashed_at"] = (k + 1) * DT
            break

        if (k + 1) in mark_steps:
            ref = read_jsbsim(fdm)
            p_ref = ned_pos(ref, lat0, lon0)
            p_our = body.pos_ned[0].cpu().numpy()
            v_our_ned = body.velocity_ned()[0].cpu().numpy()
            tot, axes = att_err_deg(body.quat[0], ref["euler"])
            out[mark_steps[k + 1]] = dict(
                pos=float(np.linalg.norm(p_our - p_ref)),
                pos_xyz=p_our - p_ref,
                vel=float(np.linalg.norm(v_our_ned - ref["vned"])),
                att=tot,
                att_axes=axes,
                rate=float(np.linalg.norm(
                    body.pqr[0].cpu().numpy() - ref["pqr"]) * DEG),
                speed_ref=float(np.linalg.norm(ref["vned"])),
                range_ref=float(np.linalg.norm(p_ref[:2])),
            )
    return out


# ---------------------------------------------------------------------------
def ballistic_check() -> None:
    """무동력·무공력 탄도 궤적.  자체 정합성 -- JSBSim 없이 해석해와 비교한다.

    외력 0 이면 남는 것은 중력뿐이고, 코리올리는 일을 하지 않으므로
    비에너지 h + V^2/(2g) 가 보존돼야 한다.

    **`gravity_varies=False` 로 돌린다.**  g 가 고도에 따라 변하면
    `h + V^2/(2g)` 는 더 이상 보존량이 아니다 (퍼텐셜이 -mu/r 로 바뀐다).
    60 s 낙하로 고도가 17 km 나 변하는 이 시험에서는 그 차이가 49 m 로 나와
    적분기 오차처럼 보인다 -- 실제로는 **지표가 틀린 것**이다.  적분기를
    재는 게 목적이므로 불변량이 정확히 성립하는 조건에서 잰다.
    """
    print("=" * 78)
    print("자체 정합성: 무동력·무공력 탄도 궤적 (60 s, 고정 중력)")
    print("=" * 78)
    for integ in ("jsbsim", "euler", "rk4"):
        for earth in (False, True):
            body = RigidBody6DOF((1,), dt=DT, dtype=DTYPE, integrator=integ,
                                 earth_rotation=earth, gravity_varies=False)
            s0 = torch.tensor([[0.0, 0.0, 6096.0, 0.0, 0.0, 0.0, 230.0, 0.0, 0.0]],
                              dtype=DTYPE)
            body.set_from_contract(s0)
            zero = torch.zeros(1, 3, dtype=DTYPE)
            w = Wrench(force=zero, moment=zero)
            mp = body.mass_props   # 이미 디바이스/정밀도에 맞춰 변환돼 있다

            h0 = 6096.0
            v0 = 230.0
            e0 = h0 + v0 * v0 / (2 * G0)
            worst_e = 0.0
            n = int(60.0 / DT)
            for _ in range(n):
                body.step(w, mp)
                c = body.to_contract()[0]
                h = float(c[2])
                v2 = float(c[6] ** 2 + c[7] ** 2 + c[8] ** 2)
                worst_e = max(worst_e, abs(h + v2 / (2 * G0) - e0))
            c = body.to_contract()[0].cpu().numpy()
            # 해석해 (평평한 지구, 코리올리 무시)
            t = 60.0
            x_a = v0 * t
            h_a = h0 - 0.5 * G0 * t * t
            vz_a = -G0 * t
            print(f"  {integ:7s} earth_rot={str(earth):5s}  "
                  f"비에너지 최대편차 {worst_e:9.3e} m   "
                  f"해석해 대비  dx={c[0] - x_a:+8.3f} m  dy={c[1]:+7.3f} m  "
                  f"dh={c[2] - h_a:+8.3f} m  dvz={c[8] - vz_a:+7.4f} m/s")
    print("  (earth_rot=True 의 해석해 편차는 코리올리 때문 -- 그게 정상이다.")
    print("   비에너지는 두 경우 모두 보존돼야 한다.)")


def contract_roundtrip() -> None:
    """계약 <-> 내부 표현 왕복이 무손실인가.  부호 실수를 여기서 잡는다."""
    print()
    print("=" * 78)
    print("계약 왕복 (ENU <-> NED/body/쿼터니언)")
    print("=" * 78)
    g = torch.Generator().manual_seed(1)
    s = torch.zeros(64, 9, dtype=DTYPE)
    s[:, 0:2] = (torch.rand(64, 2, generator=g, dtype=DTYPE) - 0.5) * 20000
    s[:, 2] = 6096.0 + (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 2000
    s[:, 3] = (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 2 * math.pi
    s[:, 4] = (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 1.2   # |theta| < 34 deg
    s[:, 5] = (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 2 * math.pi
    s[:, 6:9] = (torch.rand(64, 3, generator=g, dtype=DTYPE) - 0.5) * 400
    body = RigidBody6DOF((64,), dt=DT, dtype=DTYPE)
    body.set_from_contract(s)
    back = body.to_contract()
    d = (back - s).abs()
    d[:, 3:6] = ((back[:, 3:6] - s[:, 3:6] + math.pi) % (2 * math.pi) - math.pi).abs()
    print(f"  위치 최대오차 {float(d[:, 0:3].max()):.3e} m")
    print(f"  자세 최대오차 {math.degrees(float(d[:, 3:6].max())):.3e} deg")
    print(f"  속도 최대오차 {float(d[:, 6:9].max()):.3e} m/s")

    # ENU/NED 부호가 실제로 맞는지 못 박는 한 줄: 동쪽으로 나는 수평비행
    b2 = RigidBody6DOF((1,), dt=DT, dtype=DTYPE, earth_rotation=False)
    b2.set_from_contract(torch.tensor(
        [[0.0, 0.0, 6096.0, 0.0, 0.0, math.pi / 2, 200.0, 0.0, 0.0]], dtype=DTYPE))
    print(f"  psi=90deg, vx(동)=200 -> body uvw = "
          f"{b2.uvw[0].tolist()}  (u=200 이어야: 기수가 동쪽)")
    b3 = RigidBody6DOF((1,), dt=DT, dtype=DTYPE, earth_rotation=False)
    b3.set_from_contract(torch.tensor(
        [[0.0, 0.0, 6096.0, 0.0, 0.0, 0.0, 0.0, 200.0, 30.0]], dtype=DTYPE))
    print(f"  psi=0, vy(북)=200, vz(상승)=+30 -> uvw = "
          f"{[round(v, 3) for v in b3.uvw[0].tolist()]}  (w<0 이어야: 상승 = body z 음)")


def relative_geometry_check(seconds: float = 61.0, seed: int = 11) -> None:
    """**상대 기하** 오차.  관측이 실제로 쓰는 양이다.

    `my_observation.py` 가 보는 것은 두 기체의 절대 위치가 아니라 상대 벡터
    (거리·LOS·접근율)다.  그리고 좌표 투영 오차처럼 **두 기체에 똑같이 걸리는
    오차는 상대 벡터에서 상쇄된다.**  절대 위치 오차 50 m 가 실제로 얼마나
    나쁜지는 여기서만 답이 나온다.

    두 기체를 서로 다른 조종 입력으로 동시에 돌리고, 상대 위치 벡터의 오차와
    거리(range) 오차를 잰다.  배치 축 하나에 기체 2 대를 태운다 -- `(1, 2)`.
    """
    print()
    print("=" * 78)
    print("상대 기하 오차 (기체 2 대 동시, 배치 텐서 (1,2))")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    bes = [JSBSimF16(), JSBSimF16()]
    bes[0].reset(x=-3000.0, y=0.0, psi=0.3, v_kt=450.0)
    bes[1].reset(x=+3000.0, y=1500.0, psi=3.3, v_kt=430.0)
    fdms = [b._fdm for b in bes]
    lat0, lon0 = bes[0]._lat0, bes[0]._lon0

    for b in bes:
        b.set_controls(*FIXED_CTL)
    for _ in range(120):
        for f in fdms:
            f.run()

    body = RigidBody6DOF((1, 2), dt=DT, dtype=DTYPE)
    for i, f in enumerate(fdms):
        snap = read_jsbsim(f)
        t = lambda a: torch.as_tensor(a, dtype=DTYPE)
        body.pos_ned[0, i] = t(ned_pos(snap, lat0, lon0))
        body.uvw[0, i] = t(snap["uvw"])
        body.pqr[0, i] = t(snap["pqr"])
        e = snap["euler"]
        body.quat[0, i] = quat_from_euler(t(e[0]), t(e[1]), t(e[2]))

    n_steps = int(round(seconds / DT))
    marks = {int(round(m / DT)): m for m in (1.0, 10.0, 60.0)}
    next_change = [0, 0]
    print(f"  {'t[s]':>6}{'절대오차 평균[m]':>18}{'상대벡터오차[m]':>18}"
          f"{'거리오차[m]':>14}{'실제거리[m]':>14}{'상대오차비':>12}")
    for k in range(n_steps):
        for i in range(2):
            if k >= next_change[i]:
                # 60 s 를 버티는 기동대 -- 피치는 당기는 쪽으로만 뽑는다.
                # 전 채널 full authority 로 뽑으면 JSBSim F-16 은 45 s 안에
                # 지면에 박힌다 (우리 적분기 문제가 아니라 그게 정답이다).
                bes[i].set_controls(float(rng.uniform(-0.7, 0.7)),
                                    float(rng.uniform(0.15, 0.8)),
                                    float(rng.uniform(-0.2, 0.2)),
                                    float(rng.uniform(0.5, 1.0)))
                next_change[i] = k + int(round(float(rng.uniform(1.0, 3.0)) / DT))

        F = torch.zeros(1, 2, 3, dtype=DTYPE)
        M = torch.zeros(1, 2, 3, dtype=DTYPE)
        mv = torch.zeros(1, 2, dtype=DTYPE)
        iv = torch.zeros(4, 1, 2, dtype=DTYPE)
        for i, f in enumerate(fdms):
            s = read_jsbsim(f)
            F[0, i] = torch.as_tensor(s["force"], dtype=DTYPE)
            M[0, i] = torch.as_tensor(s["moment"], dtype=DTYPE)
            mv[0, i] = s["mass"]
            for j, key in enumerate(("ixx", "iyy", "izz", "ixz")):
                iv[j, 0, i] = s[key]
        body.step(Wrench(force=F, moment=M),
                  MassProps(mass=mv, ixx=iv[0], iyy=iv[1], izz=iv[2], ixz=iv[3]))
        for f in fdms:
            f.run()

        if any(not (0.0 < f["position/h-sl-ft"] < 100000.0) for f in fdms):
            print(f"  (JSBSim 이 {(k + 1) * DT:.1f} s 에 지면 충돌)")
            break
        if (k + 1) in marks:
            refs = [ned_pos(read_jsbsim(f), lat0, lon0) for f in fdms]
            ours = body.pos_ned[0].cpu().numpy()
            abs_e = float(np.mean([np.linalg.norm(ours[i] - refs[i]) for i in (0, 1)]))
            rel_ref = refs[1] - refs[0]
            rel_our = ours[1] - ours[0]
            rel_e = float(np.linalg.norm(rel_our - rel_ref))
            rng_e = abs(float(np.linalg.norm(rel_our) - np.linalg.norm(rel_ref)))
            R = float(np.linalg.norm(rel_ref))
            print(f"  {marks[k + 1]:>6.0f}{abs_e:>18.3f}{rel_e:>18.3f}"
                  f"{rng_e:>14.3f}{R:>14.0f}{rel_e / max(abs_e, 1e-9):>12.2f}")
    print("  (상대오차비 < 1 이면 공통 오차가 상쇄됐다는 뜻 -- 관측이 보는 것은 이 쪽이다)")


def main() -> None:
    torch.set_printoptions(precision=6)
    contract_roundtrip()
    print()
    ballistic_check()

    print()
    print("=" * 78)
    print("JSBSim 대조 -- 힘 주입 (적분기 오차만 분리)")
    print("=" * 78)
    header = (f"  {'적분기':<10}{'지구항':<14}{'t[s]':>6}{'위치[m]':>12}"
              f"{'속도[m/s]':>11}{'자세[deg]':>11}{'각속도[deg/s]':>14}")
    # (earth_rotation, frame_correction, 이름)
    variants = [(True, True, "코리올리+보정"), (True, False, "코리올리만"),
                (False, False, "완전평지구")]
    for mode, label in (
            ("fixed", f"고정 입력 {FIXED_CTL} -- 연속 루프, 60 s 생존"),
            ("random", "무작위 입력 (1~3 s 마다 4채널 전부 재추첨)")):
        print()
        print(f"  ## {label}")
        print(header)
        for integ in ("euler", "jsbsim", "rk4"):
            for earth, fc, vname in variants:
                if integ != "jsbsim" and not earth:
                    continue          # 지구항 절제는 대표 적분기 하나로만
                r = run_case(integ, 61.0, mode, seed=7,
                             earth_rotation=earth, frame_correction=fc)
                for t in (1.0, 10.0, 60.0):
                    if t not in r:
                        print(f"  {integ:<10}{vname:<14}{t:>6.0f}"
                              f"{'  -- 기체 소실 --':>40}")
                        continue
                    e = r[t]
                    print(f"  {integ:<10}{vname:<14}{t:>6.0f}"
                          f"{e['pos']:>12.4f}{e['vel']:>11.5f}"
                          f"{e['att']:>11.5f}{e['rate']:>14.5f}")
                if "crashed_at" in r:
                    print(f"      (JSBSim 이 {r['crashed_at']:.1f} s 에 지면 충돌)")

    # 축별 분해 -- 부호 실수는 축 하나에만 나타난다
    print()
    print("  ## 축별 분해 (jsbsim 적분기, 무작위 입력, 60 s)")
    r = run_case("jsbsim", 61.0, "random", seed=7)
    for t in (1.0, 10.0, 60.0):
        if t in r:
            e = r[t]
            print(f"    t={t:>4.0f}s  위치NED=({e['pos_xyz'][0]:+8.3f},"
                  f"{e['pos_xyz'][1]:+8.3f},{e['pos_xyz'][2]:+8.3f}) m   "
                  f"자세(phi,th,psi)=({e['att_axes'][0]:+7.4f},"
                  f"{e['att_axes'][1]:+7.4f},{e['att_axes'][2]:+7.4f}) deg   "
                  f"기준속도 {e['speed_ref']:.1f} m/s  기준거리 {e['range_ref']:.0f} m")

    relative_geometry_check()


if __name__ == "__main__":
    main()
