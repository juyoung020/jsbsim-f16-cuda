# SPDX-License-Identifier: GPL-3.0-or-later
"""F16Stick 을 CUDA 커널 하나로 -- 기체 하나 = 스레드 하나.

    dyn = f16_core.F16Stick(65536, device="cuda")
    fused_core.attach_stick(dyn)               # 이제 dyn.step(stick, substeps) 가 커널로 돈다

한 프레임의 계산(보조량 -> FLCS -> F100 터빈 -> 공력 -> 6-DOF -> 질량·연료)을 torch 판
`f16_core.F16Stick._substep` 과 **같은 식·같은 순서**로 CUDA C++ 로 옮겼다.  공력 계수표와
상수는 torch 모듈(`aero`, `propulsion`, `flcs`, `rbdyn`, `TankMass`)에서 **읽어 소스를
생성한다** -- 숫자를 두 군데 두지 않는다.  상태를 레지스터에 올린 채 `substeps` 프레임을
돌고 끝에 한 번 쓴다.  상태 텐서는 F16Stick 이 들고 있는 것을 **제자리에서** 읽고 쓰므로
`reset`, `state()`, 검증 도구가 그대로 돈다.

**NVRTC** 로 런타임 컴파일한다 (`nvrtc.py`, torch 가 들고 오는 라이브러리) -- CUDA 툴킷도
C++ 컴파일러도 필요 없다.  처음 한 번 컴파일하고 결과(CUBIN)를 디스크에 캐시한다.
`REAL=float` 와 `REAL=double` 로 따로 컴파일된다.

torch 판과 **비트까지** 같지는 않다 (torch 는 연산마다 결과를 한 번씩 반올림하고, 공력 축
합산이 행렬곱이다).  float64 로 대조하면 상대 1e-13 안이다.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch


from jsbsim_f16_cuda import flcs as FL                                 # noqa: E402
from jsbsim_f16_cuda import propulsion as PR                           # noqa: E402
from jsbsim_f16_cuda import rbdyn as RB                                # noqa: E402

#: 검증용 중간량 (`fdm_verify` 의 키 이름).  힘·FCS·보조량은 그 프레임의 것, 뒤 13 개는
#: 적분 뒤 상태다.
DBG_KEYS = (
    "alpha", "beta", "mach", "qbar", "vc", "vg", "npz", "npy",
    "ele_pos", "ail_pos", "rud_pos", "lef_pos", "sb_pos", "flap_mix",
    "thrust", "n2", "fuel", "mass", "ixx", "iyy", "izz", "ixz", "cgx", "cgz",
    "fbx_a", "fby_a", "fbz_a", "l_a", "m_a", "n_a",
    "fbx", "fby", "fbz", "l", "m", "n",
    "udot", "vdot", "wdot", "pdot", "qdot", "rdot",
    "u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "h", "vn", "ve", "vd",
)
NDBG = len(DBG_KEYS)

# 단위 (f16_core 와 같은 값)
FT = 0.3048
KT_TO_MS = 0.514444
FPS_PER_KT = 1.6878098571
P_SL_PSF = 2116.228
R_ENG = 1716.55716
RHO_SL = P_SL_PSF / (R_ENG * 518.67)
A_SL_FPS = math.sqrt(1.4 * P_SL_PSF / RHO_SL)
G_STD_FPS2 = 32.174049
EYEPOINT_IN = (-336.2, 0.0, 29.5)
DT_PHYS = 1.0 / 120.0
FF_RATE_UP_PPH_S = 5000.0
FF_RATE_DOWN_PPH_S = 10000.0


# =============================================================================
# 상수 출력 -- dtype 별로 **그 dtype 에서 정확히 같은 값**이 되게 적는다
# =============================================================================

def lit(x: float, dbl: bool) -> str:
    """float32 면 float32 로 반올림한 값을 9 자리로 (문자열 -> float 가 같은 값),
    float64 면 repr.  torch 가 파이썬 실수 상수를 텐서 dtype 으로 내리는 것과 같다."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(x)
    if dbl:
        s = repr(x)
        if "e" not in s and "." not in s:
            s += ".0"
        return s
    return f"{float(np.float32(x)):.9e}f"


def arr(name: str, vals, dbl: bool) -> str:
    vals = [float(v) for v in np.asarray(vals, dtype=np.float64).ravel()]
    return (f"__device__ const REAL {name}[{len(vals)}] = "
            f"{{{', '.join(lit(v, dbl) for v in vals)}}};\n")


def defines(c: dict, dbl: bool) -> str:
    return "".join(f"#define {k} ({lit(v, dbl)})\n" for k, v in c.items())


def _core_consts(dbl: bool) -> str:
    c = dict(
        FT=FT, KT_TO_MS=KT_TO_MS, LBF=RB.LBF, LBFFT=RB.LBFFT, SLUG=RB.SLUG,
        SLUGFT2=RB.SLUGFT2, DT=DT_PHYS, INV_DT=1.0 / DT_PHYS,
        P_SL_PSF=P_SL_PSF, G_STD=G_STD_FPS2, VC_SCALE=A_SL_FPS / FPS_PER_KT,
        EYE_X=EYEPOINT_IN[0], EYE_Y=EYEPOINT_IN[1], EYE_Z=EYEPOINT_IN[2],
        FF_UP=FF_RATE_UP_PPH_S * DT_PHYS / 3600.0,
        FF_DN=FF_RATE_DOWN_PPH_S * DT_PHYS / 3600.0,
        R_EARTH=RB.R_EARTH, RH_REF=RB.R_EARTH + RB.H_REF,
        AB3_0=23.0 / 12.0, AB3_1=16.0 / 12.0, AB3_2=5.0 / 12.0,
        ATM_EARTH_R=PR._EARTH_R, ATM_G0=PR._G0, ATM_R=PR._R_ENG, ATM_GAMMA=PR._GAMMA,
        RHO_SL_TURB=PR._RHO_SL,
        PI=math.pi, TWO_PI=2.0 * math.pi, DEG2RAD=math.pi / 180.0,
        ROLL_RATE_GAIN=FL.ROLL_RATE_GAIN, ROLL_KP=FL.ROLL_KP, ROLL_KD=FL.ROLL_KD,
        AIL_RANGE=FL.AILERON_RANGE_RAD, AIL_RATE=FL.AILERON_RATE * DT_PHYS,
        FLAP_MIX=FL.FLAPERON_MIX_GAIN,
        PITCH_KP=FL.PITCH_KP, PITCH_KD=FL.PITCH_KD,
        PITCH_RATE_GAIN=FL.PITCH_RATE_GAIN, G_LOAD_GAIN=FL.G_LOAD_GAIN,
        ALPHA_LIM_GAIN=FL.ALPHA_LIMITER_GAIN, ELEV_MIN=FL.ELEV_CMD_MIN,
        ELEV_MAX=FL.ELEV_CMD_MAX, ELE_RANGE=FL.ELEVATOR_RANGE_RAD,
        ELE_RATE=FL.ELEVATOR_RATE * DT_PHYS,
        YAW_KP=FL.YAW_KP, YAW_KD=FL.YAW_KD, YAW_LOAD_GAIN=FL.YAW_LOAD_GAIN,
        RUD_RANGE=FL.RUDDER_RANGE_RAD, RUD_RATE=FL.RUDDER_RATE * DT_PHYS,
        TEF_NORM_GAIN=FL.TEF_NORM_GAIN, TEF_RATE=FL.TEF_RATE * DT_PHYS,
        TEF_VC=FL.TEF_VC_KT, TEF_HI_MACH=FL.TEF_HI_MACH_RAD, TEF_LO=0.349,
        LEF_ALPHA_LO=FL.LEF_ALPHA_LO, LEF_LO=0.262, LEF_HI_MACH=FL.LEF_HI_MACH_RAD,
        MACH_HI=FL.MACH_HI,
        SCHED_A1=FL.SCHED_A1, SCHED_A2=FL.SCHED_A2, SCHED_SLOPE1=FL.SCHED_SLOPE1,
        SCHED_Y1=FL.SCHED_Y1, SCHED_INV=1.0 / (FL.SCHED_A2 - FL.SCHED_A1),
        VG_INV20=1.0 / 20.0, VG_INV50=1.0 / 50.0,
    )
    return defines(c, dbl)


def _aero_code(aero, dbl: bool) -> tuple[str, str]:
    """(전역 표 선언, 커널 안 계산 코드).  `F16Aero.functions` + 축 합산.

    독립변수마다 구간 한 번, 함수마다 `prod(X[gather]) * TV[tv] * const`.  **합산
    순서는 함수 번호 순**이다 (torch 는 `vals @ axis_mat` GEMM 이라 구현 정의다).
    """
    decl, code = [], []
    tv_expr = {0: "one"}
    st = ["one"] + [f"X_{n}" for n in aero.state_names]
    for i, (var, bp, data, idx) in enumerate(aero._g1):
        decl.append(arr(f"A1BP{i}", bp, dbl))
        decl.append(arr(f"A1D{i}", data, dbl))
        n = len(bp)
        code.append(f"    int a1i{i}; REAL a1t{i}; bracket(A1BP{i}, {n}, X_{var}, a1i{i}, a1t{i});\n")
        for k, slot in enumerate(idx):
            code.append(f"    REAL tv{slot} = lerp_t(A1D{i}[{k * n} + a1i{i}], "
                        f"A1D{i}[{k * n} + a1i{i} + 1], a1t{i});\n")
            tv_expr[slot] = f"tv{slot}"
    for i, (rv, cv, rbp, cbp, data, idx) in enumerate(aero._g2):
        decl.append(arr(f"A2R{i}", rbp, dbl))
        decl.append(arr(f"A2C{i}", cbp, dbl))
        decl.append(arr(f"A2D{i}", data, dbl))
        R, Cn = len(rbp), len(cbp)
        code.append(f"    int a2r{i}, a2c{i}; REAL a2rt{i}, a2ct{i};\n"
                    f"    bracket(A2R{i}, {R}, X_{rv}, a2r{i}, a2rt{i});\n"
                    f"    bracket(A2C{i}, {Cn}, X_{cv}, a2c{i}, a2ct{i});\n")
        for k, slot in enumerate(idx):
            b = f"{k * R * Cn} + a2r{i} * {Cn} + a2c{i}"
            code.append(f"    REAL tv{slot} = interp2(A2D{i}, {b}, {Cn}, a2rt{i}, a2ct{i});\n")
            tv_expr[slot] = f"tv{slot}"
    gidx = aero._gather_idx.cpu().numpy()
    tvof = aero._tv_of_fn.cpu().numpy()
    const = aero._const.double().cpu().numpy()
    axis = aero._axis_mat.double().cpu().numpy().argmax(1)
    sums = {a: [] for a in range(6)}
    for f in range(len(tvof)):
        prod = st[int(gidx[f, 0])]
        for k in range(1, gidx.shape[1]):
            prod = f"({prod} * {st[int(gidx[f, k])]})"
        code.append(f"    REAL fn{f} = ({prod} * {tv_expr[int(tvof[f])]}) * {lit(const[f], dbl)};\n")
        sums[int(axis[f])].append(f"fn{f}")
    names = ("aD", "aY", "aL", "al", "am", "an")
    for a in range(6):
        expr = sums[a][0]
        for t in sums[a][1:]:
            expr = f"({expr} + {t})"
        code.append(f"    REAL {names[a]} = {expr};\n")
    return "".join(decl), "".join(code)


def _mass_tables(mass, dbl: bool) -> str:
    """질량 모델.  표(`MassModel`, 급유 램프) 또는 탱크 네 개(`TankMass`)."""
    if hasattr(mass, "_TAB"):                                   # MassModel (표)
        return ("#define MASS_TANKS 0\n"
                + arr("MS_F", mass._F, dbl) + arr("MS_T", mass._TAB, dbl)
                + f"#define MS_N {len(mass._F)}\n"
                + defines(dict(REFUEL_PPS=mass.REFUEL_PPS, FUEL_CAP=mass.FUEL_CAP_LBS,
                               EMPTY_LBS=mass.EMPTY_LBS), dbl))
    # TankMass -- 상수는 인스턴스에서 읽는다 (클래스 속성 그대로)
    if len(mass.POINTMASSES) != 1:
        raise ValueError("fused_core: 점질량은 조종사 하나만 옮겼다")
    (pw, pr), = mass.POINTMASSES
    fixed_mom = [mass.EMPTY_LBS * mass.CG_EMPTY_IN[i] + pw * pr[i] for i in range(3)]
    return ("#define MASS_TANKS 1\n"
            + f"#define TK_REFUEL {1 if mass.refuel else 0}\n"
            + arr("TK_R", mass.TANK_IN, dbl) + arr("TK_CAP", mass.TANK_CAP_LBS, dbl)
            + arr("TK_MOM", fixed_mom, dbl)
            + defines(dict(TK_WFIX=mass.EMPTY_LBS + pw,
                           TK_JXX=mass.J_EMPTY[0], TK_JYY=mass.J_EMPTY[1],
                           TK_JZZ=mass.J_EMPTY[2], TK_JXZ=mass.J_EMPTY[3],
                           TK_ME=mass.EMPTY_LBS / G_STD_FPS2, TK_MP=pw / G_STD_FPS2,
                           TK_CGE_X=mass.CG_EMPTY_IN[0], TK_CGE_Y=mass.CG_EMPTY_IN[1],
                           TK_CGE_Z=mass.CG_EMPTY_IN[2],
                           TK_PR_X=pr[0], TK_PR_Y=pr[1], TK_PR_Z=pr[2],
                           TK_AX=-1.0 / 12.0, TK_AY=1.0 / 12.0, TK_AZ=-1.0 / 12.0,
                           TK_REFUEL_STEP=mass.REFUEL_PPS * DT_PHYS), dbl))


def _tables(aero, turb, dbl: bool) -> str:
    out = []
    out.append(arr("KGE_BP", aero._kge_bp.double().cpu().numpy(), dbl))
    out.append(arr("KGE_D", aero._kge_data.double().cpu().numpy()[0], dbl))
    out.append(f"#define KGE_N {aero._kge_bp.numel()}\n")
    out.append(arr("TB_R", turb._rbp.double().cpu().numpy(), dbl))
    out.append(arr("TB_C", turb._cbp.double().cpu().numpy(), dbl))
    out.append(arr("TB_D", turb._data.double().cpu().numpy(), dbl))
    out.append(f"#define TB_NR {turb._rbp.numel()}\n#define TB_NC {turb._cbp.numel()}\n")
    out.append(defines(dict(TB_MIL=turb.mil, TB_MAX=turb.max, TB_IDLE=turb.idle_n2,
                            TB_FACTOR=turb.n2_factor, TB_UP=turb.spool_up,
                            TB_DN=turb.spool_dn), dbl))
    out.append(arr("ATM_H0", PR._H_GEOPOT, dbl))
    out.append(arr("ATM_T0", PR._T_BASE, dbl))
    out.append(arr("ATM_LAM", PR._LAPSE, dbl))
    out.append(arr("ATM_P0", PR._P_BASE, dbl))
    out.append(f"#define ATM_N {len(PR._H_GEOPOT)}\n")
    rp = aero.aerorp_in
    out.append(defines(dict(AERO_BW=aero.metrics["bw_ft"], AERO_CBAR=aero.metrics["cbar_ft"],
                            RP_X=rp[0], RP_Y=rp[1], RP_Z=rp[2]), dbl))
    return "".join(out)


_MATH = {
    False: dict(REAL="float", SQRT="sqrtf", ATAN2="atan2f", ASIN="asinf", ACOS="acosf",
                SIN="sinf", COS="cosf", EXP="expf", POW="powf", FMOD="fmodf", ABS="fabsf",
                FLOOR="floorf", HYPOT="hypotf", ISFINITE="isfinite"),
    True: dict(REAL="double", SQRT="sqrt", ATAN2="atan2", ASIN="asin", ACOS="acos",
               SIN="sin", COS="cos", EXP="exp", POW="pow", FMOD="fmod", ABS="fabs",
               FLOOR="floor", HYPOT="hypot", ISFINITE="isfinite"),
}


# =============================================================================
# 코어 -- 한 프레임 = core_pre (질량·보조량) + [드라이버가 조종간을 정한다] + core_post
# =============================================================================

_CORE = r"""
// ---------------------------------------------------------------- 수학 도우미
// torch 의 clamp / minimum / maximum 은 NaN 을 **통과시킨다**.  fminf 는 NaN 을
// 삼키므로 쓰지 않는다.
__device__ __forceinline__ REAL cl(REAL x, REAL lo, REAL hi) {
    return (x != x) ? x : (x < lo ? lo : (x > hi ? hi : x));
}
__device__ __forceinline__ REAL cmin(REAL x, REAL lo) { return (x != x) ? x : (x < lo ? lo : x); }  // clamp_min
__device__ __forceinline__ REAL cmax(REAL x, REAL hi) { return (x != x) ? x : (x > hi ? hi : x); }  // clamp_max
__device__ __forceinline__ REAL tmin(REAL a, REAL b) {  // torch.minimum
    return (a != a) ? a : ((b != b) ? b : (a < b ? a : b));
}
__device__ __forceinline__ REAL tmax(REAL a, REAL b) {  // torch.maximum
    return (a != a) ? a : ((b != b) ? b : (a > b ? a : b));
}
// torch.remainder (부동소수): fmod 뒤 부호 보정 -- 파이썬 `%` 규약
__device__ __forceinline__ REAL rem(REAL a, REAL b) {
    REAL m = FMOD(a, b);
    if ((m != 0) && ((b < 0) != (m < 0))) m += b;
    return m;
}
__device__ __forceinline__ REAL wrap_pi(REAL a) { return rem(a + PI, TWO_PI) - PI; }
// torch.lerp
__device__ __forceinline__ REAL tlerp(REAL s, REAL e, REAL w) {
    return (ABS(w) < (REAL)0.5) ? s + w * (e - s) : e - (e - s) * ((REAL)1 - w);
}
// addcmul(lo, hi - lo, t)
__device__ __forceinline__ REAL lerp_t(REAL lo, REAL hi, REAL t) { return lo + t * (hi - lo); }

// f16_interp.bracket: searchsorted(right=False) = 첫 bp >= x, [1, n-1] 로 물림
__device__ __forceinline__ void bracket(const REAL* bp, int n, REAL x, int& i0, REAL& t) {
    int i1 = 0;
    while (i1 < n && bp[i1] < x) ++i1;         // NaN 이면 n -> 물림
    if (i1 < 1) i1 = 1;
    if (i1 > n - 1) i1 = n - 1;
    i0 = i1 - 1;
    REAL lo = bp[i0];
    t = cl((x - lo) / (bp[i1] - lo), (REAL)0, (REAL)1);
}
__device__ __forceinline__ REAL interp2(const REAL* d, int base, int C, REAL rt, REAL ct) {
    REAL g00 = d[base], g01 = d[base + 1], g10 = d[base + C], g11 = d[base + C + 1];
    REAL c0 = g00 + rt * (g10 - g00);
    REAL c1 = g01 + rt * (g11 - g01);
    return c0 + ct * (c1 - c0);
}

// ---------------------------------------------------------------- 쿼터니언
struct Dcm { REAL t[9]; };
__device__ __forceinline__ Dcm dcm_l2b(REAL w, REAL x, REAL y, REAL z) {
    Dcm d;
    REAL w2 = w * w, x2 = x * x, y2 = y * y, z2 = z * z;
    d.t[0] = w2 + x2 - y2 - z2; d.t[1] = 2 * (x * y + w * z); d.t[2] = 2 * (x * z - w * y);
    d.t[3] = 2 * (x * y - w * z); d.t[4] = w2 - x2 + y2 - z2; d.t[5] = 2 * (y * z + w * x);
    d.t[6] = 2 * (x * z + w * y); d.t[7] = 2 * (y * z - w * x); d.t[8] = w2 - x2 - y2 + z2;
    return d;
}
__device__ __forceinline__ void euler(REAL w, REAL x, REAL y, REAL z, REAL& phi, REAL& th, REAL& psi) {
    REAL w2 = w * w, x2 = x * x, y2 = y * y, z2 = z * z;
    REAL m02 = 2 * (x * z - w * y);
    phi = ATAN2(2 * (y * z + w * x), w2 - x2 - y2 + z2);
    th = ASIN(cl(-m02, (REAL)-1, (REAL)1));
    psi = ATAN2(2 * (x * y + w * z), w2 + x2 - y2 - z2);
}

// ---------------------------------------------------------------- 대기
__device__ __forceinline__ void atmos(REAL h_ft, REAL& T, REAL& P, REAL& rho, REAL& a) {
    REAL hg = h_ft * ATM_EARTH_R / (ATM_EARTH_R + h_ft);
    int i = 0;                                   // searchsorted(right=True) - 1
    while (i < ATM_N && ATM_H0[i] <= hg) ++i;
    i -= 1;
    if (i < 0) i = 0;
    if (i > ATM_N - 1) i = ATM_N - 1;
    REAL h0 = ATM_H0[i], T0 = ATM_T0[i], lam = ATM_LAM[i], P0 = ATM_P0[i];
    REAL dh = hg - h0;
    T = T0 + lam * dh;
    if (ABS(lam) < (REAL)1e-12)
        P = P0 * EXP(-ATM_G0 * dh / (ATM_R * T0));
    else
        P = P0 * POW(T / T0, -ATM_G0 / (ATM_R * lam));
    rho = P / (ATM_R * T);
    a = SQRT(ATM_GAMMA * ATM_R * T);
}

__device__ __forceinline__ REAL vcal_kts(REAL mach, REAL p) {
    // pitot_total_pressure
    REAL ms = cmin(mach, (REAL)1e-6);
    REAL pt;
    if (mach < (REAL)1) {
        pt = p * POW((REAL)1 + (REAL)0.2 * mach * mach, (REAL)3.5);
    } else {
        pt = p * (REAL)166.92158009316827 * POW(ms, (REAL)7)
             / POW(cmin((REAL)7 * ms * ms - (REAL)1, (REAL)1e-9), (REAL)2.5);
    }
    // mach_from_impact_pressure(pt - p, P_SL)
    REAL qc = pt - p;
    REAL A = cmin(qc / P_SL_PSF + (REAL)1, (REAL)1e-9);
    REAL m;
    if (A < (REAL)1.89293) {
        m = SQRT(cmin((REAL)5 * (POW(A, (REAL)(2.0 / 7.0)) - (REAL)1), (REAL)0));
    } else {
        m = (REAL)2;
        #pragma unroll 1
        for (int k = 0; k < 12; ++k)
            m = (REAL)0.881285 * SQRT(A * POW(cmin((REAL)1 - (REAL)1 / ((REAL)7 * m * m), (REAL)1e-9), (REAL)2.5));
    }
    return m * VC_SCALE;
}

// ---------------------------------------------------------------- 추진
__device__ __forceinline__ REAL turb_thrust(REAL n2norm, REAL aug, REAL mach, REAL alt) {
    int ri, ci; REAL rt, ct;
    bracket(TB_R, TB_NR, mach, ri, rt);
    bracket(TB_C, TB_NC, alt, ci, ct);
    int b = ri * TB_NC + ci;
    REAL tv0 = interp2(TB_D, b, TB_NC, rt, ct);
    REAL tv1 = interp2(TB_D, TB_NR * TB_NC + b, TB_NC, rt, ct);
    REAL tv2 = interp2(TB_D, 2 * TB_NR * TB_NC + b, TB_NC, rt, ct);
    REAL idle = TB_MIL * tv0;
    REAL milt = (TB_MIL - idle) * tv1;
    REAL t = idle + milt * n2norm * n2norm;
    REAL tdiff = TB_MAX * tv2 - t;
    return (aug > (REAL)0) ? t + tdiff * cmax(aug, (REAL)1) : t;
}
__device__ __forceinline__ REAL fuel_flow(REAL thrust, REAL aug, REAL n2norm) {
    REAL sfc = (aug > (REAL)0) ? (REAL)2.05
             : (REAL)0.666 * ((REAL)1 + (REAL)1.1907 * ((REAL)1 - n2norm) * ((REAL)1 - n2norm));
    return cmin(cmin(thrust, (REAL)0) * sfc, (REAL)756) / (REAL)3600;
}

// ---------------------------------------------------------------- 공력 표
@@AERO_DECL@@

// ---------------------------------------------------------------- 상태
// 스텝을 건너 사는 것 전부.  레지스터에 올려 두고 드라이버가 마지막에 한 번 쓴다.
struct Core {
    REAL pn, pe, pd, u, v, w, qw, qx, qy, qz, p, q, r;
    bool fr;
    REAL hu0x, hu0y, hu0z, hu1x, hu1y, hu1z, hu2x, hu2y, hu2z;
    REAL hp0x, hp0y, hp0z, hp1x, hp1y, hp1z, hp2x, hp2y, hp2z;
    REAL ail_pn, ele_pn, tef, i_roll, e_roll_p, i_pitch, e_pitch_p, i_yaw, e_yaw_p;
    REAL d_alpha, d_p, d_q, d_r, d_nz, d_ny, d_mach, d_vc, d_vg;
    REAL n2, n2norm, ff_pps, ptrim;
    REAL abx, aby, abz, wdx, wdy, wdz, npx, npy_, npz_;
#if MASS_TANKS
    REAL f0, f1, f2, f3, cgtx, cgty, cgtz;
#else
    REAL fuel;
#endif
};
// 한 프레임의 앞부분이 뒷부분에 넘기는 값
struct Pre {
    REAL mass_slugs, m_kg, Ixx, Iyy, Izz, Ixz, cgx, cgy, cgz;
    REAL jxx, jyy, jzz, jxz, fuel_tot;               // 검증용 (slug ft^2, lb)
    REAL phi, theta, psi, h_m, h_ft, vt_ms, vt_fps, alpha, beta, rho, mach, qbar, vc_kts;
    Dcm T;
    REAL vn, ve, vd, vg_fps, h_dot_ms;
    REAL web_x, web_y, web_z, wix, wiy, wiz;
};
// 위도·중력 (인스턴스마다 다르다 -- 커널 인자로 받는다)
struct Env { REAL G0, WE_N, WE_D, ACENT_D; };

// 상태 포인터 -- torch 플랜트의 텐서들 (제자리에서 읽고 쓴다)
struct CorePtr {
    REAL *pos, *uvw, *quat, *pqr; unsigned char* fresh; REAL *hu, *hp, *fst;
    REAL *n2, *n2n, *fuel, *ff; const REAL* ptrim;
    REAL *ab, *wd, *np; REAL* cgt;
};

__device__ __forceinline__ void core_load(Core& c, const CorePtr& P, int i, int N) {
    c.pn = P.pos[3 * i]; c.pe = P.pos[3 * i + 1]; c.pd = P.pos[3 * i + 2];
    c.u = P.uvw[3 * i]; c.v = P.uvw[3 * i + 1]; c.w = P.uvw[3 * i + 2];
    c.qw = P.quat[4 * i]; c.qx = P.quat[4 * i + 1]; c.qy = P.quat[4 * i + 2]; c.qz = P.quat[4 * i + 3];
    c.p = P.pqr[3 * i]; c.q = P.pqr[3 * i + 1]; c.r = P.pqr[3 * i + 2];
    c.fr = P.fresh[i] != 0;
    const int N3 = 3 * N;
    const REAL* hu = P.hu; const REAL* hp = P.hp;
    c.hu0x = hu[3 * i]; c.hu0y = hu[3 * i + 1]; c.hu0z = hu[3 * i + 2];
    c.hu1x = hu[N3 + 3 * i]; c.hu1y = hu[N3 + 3 * i + 1]; c.hu1z = hu[N3 + 3 * i + 2];
    c.hu2x = hu[2 * N3 + 3 * i]; c.hu2y = hu[2 * N3 + 3 * i + 1]; c.hu2z = hu[2 * N3 + 3 * i + 2];
    c.hp0x = hp[3 * i]; c.hp0y = hp[3 * i + 1]; c.hp0z = hp[3 * i + 2];
    c.hp1x = hp[N3 + 3 * i]; c.hp1y = hp[N3 + 3 * i + 1]; c.hp1z = hp[N3 + 3 * i + 2];
    c.hp2x = hp[2 * N3 + 3 * i]; c.hp2y = hp[2 * N3 + 3 * i + 1]; c.hp2z = hp[2 * N3 + 3 * i + 2];
    const REAL* f = P.fst;                        // BatchFLCS.N_STATE_KEYS 순서
    c.ail_pn = f[0 * N + i]; c.ele_pn = f[1 * N + i]; c.tef = f[2 * N + i];
    c.i_roll = f[3 * N + i]; c.e_roll_p = f[4 * N + i];
    c.i_pitch = f[5 * N + i]; c.e_pitch_p = f[6 * N + i];
    c.i_yaw = f[7 * N + i]; c.e_yaw_p = f[8 * N + i];
    c.d_alpha = f[9 * N + i]; c.d_p = f[10 * N + i]; c.d_q = f[11 * N + i]; c.d_r = f[12 * N + i];
    c.d_nz = f[13 * N + i]; c.d_ny = f[14 * N + i];
    c.d_mach = f[15 * N + i]; c.d_vc = f[16 * N + i]; c.d_vg = f[17 * N + i];
    c.n2 = P.n2[i]; c.n2norm = P.n2n[i]; c.ff_pps = P.ff[i]; c.ptrim = P.ptrim[i];
    c.abx = P.ab[3 * i]; c.aby = P.ab[3 * i + 1]; c.abz = P.ab[3 * i + 2];
    c.wdx = P.wd[3 * i]; c.wdy = P.wd[3 * i + 1]; c.wdz = P.wd[3 * i + 2];
    c.npx = P.np[3 * i]; c.npy_ = P.np[3 * i + 1]; c.npz_ = P.np[3 * i + 2];
#if MASS_TANKS
    c.f0 = P.fuel[4 * i]; c.f1 = P.fuel[4 * i + 1]; c.f2 = P.fuel[4 * i + 2]; c.f3 = P.fuel[4 * i + 3];
    c.cgtx = P.cgt[3 * i]; c.cgty = P.cgt[3 * i + 1]; c.cgtz = P.cgt[3 * i + 2];
#else
    c.fuel = P.fuel[i];
#endif
}

__device__ __forceinline__ void core_store(const Core& c, const CorePtr& P, int i, int N) {
    P.pos[3 * i] = c.pn; P.pos[3 * i + 1] = c.pe; P.pos[3 * i + 2] = c.pd;
    P.uvw[3 * i] = c.u; P.uvw[3 * i + 1] = c.v; P.uvw[3 * i + 2] = c.w;
    P.quat[4 * i] = c.qw; P.quat[4 * i + 1] = c.qx; P.quat[4 * i + 2] = c.qy; P.quat[4 * i + 3] = c.qz;
    P.pqr[3 * i] = c.p; P.pqr[3 * i + 1] = c.q; P.pqr[3 * i + 2] = c.r;
    P.fresh[i] = c.fr ? 1 : 0;
    const int N3 = 3 * N;
    REAL* hu = P.hu; REAL* hp = P.hp;
    hu[3 * i] = c.hu0x; hu[3 * i + 1] = c.hu0y; hu[3 * i + 2] = c.hu0z;
    hu[N3 + 3 * i] = c.hu1x; hu[N3 + 3 * i + 1] = c.hu1y; hu[N3 + 3 * i + 2] = c.hu1z;
    hu[2 * N3 + 3 * i] = c.hu2x; hu[2 * N3 + 3 * i + 1] = c.hu2y; hu[2 * N3 + 3 * i + 2] = c.hu2z;
    hp[3 * i] = c.hp0x; hp[3 * i + 1] = c.hp0y; hp[3 * i + 2] = c.hp0z;
    hp[N3 + 3 * i] = c.hp1x; hp[N3 + 3 * i + 1] = c.hp1y; hp[N3 + 3 * i + 2] = c.hp1z;
    hp[2 * N3 + 3 * i] = c.hp2x; hp[2 * N3 + 3 * i + 1] = c.hp2y; hp[2 * N3 + 3 * i + 2] = c.hp2z;
    REAL* f = P.fst;
    f[0 * N + i] = c.ail_pn; f[1 * N + i] = c.ele_pn; f[2 * N + i] = c.tef;
    f[3 * N + i] = c.i_roll; f[4 * N + i] = c.e_roll_p;
    f[5 * N + i] = c.i_pitch; f[6 * N + i] = c.e_pitch_p;
    f[7 * N + i] = c.i_yaw; f[8 * N + i] = c.e_yaw_p;
    f[9 * N + i] = c.d_alpha; f[10 * N + i] = c.d_p; f[11 * N + i] = c.d_q; f[12 * N + i] = c.d_r;
    f[13 * N + i] = c.d_nz; f[14 * N + i] = c.d_ny;
    f[15 * N + i] = c.d_mach; f[16 * N + i] = c.d_vc; f[17 * N + i] = c.d_vg;
    P.n2[i] = c.n2; P.n2n[i] = c.n2norm; P.ff[i] = c.ff_pps;
    P.ab[3 * i] = c.abx; P.ab[3 * i + 1] = c.aby; P.ab[3 * i + 2] = c.abz;
    P.wd[3 * i] = c.wdx; P.wd[3 * i + 1] = c.wdy; P.wd[3 * i + 2] = c.wdz;
    P.np[3 * i] = c.npx; P.np[3 * i + 1] = c.npy_; P.np[3 * i + 2] = c.npz_;
#if MASS_TANKS
    P.fuel[4 * i] = c.f0; P.fuel[4 * i + 1] = c.f1; P.fuel[4 * i + 2] = c.f2; P.fuel[4 * i + 3] = c.f3;
    P.cgt[3 * i] = c.cgtx; P.cgt[3 * i + 1] = c.cgty; P.cgt[3 * i + 2] = c.cgtz;
#else
    P.fuel[i] = c.fuel;
#endif
}

// 계약 (ENU 9 값)
__device__ __forceinline__ void core_contract(const Core& c, REAL* o) {
    REAL phi, theta, psi;
    euler(c.qw, c.qx, c.qy, c.qz, phi, theta, psi);
    Dcm T = dcm_l2b(c.qw, c.qx, c.qy, c.qz);
    REAL vn = T.t[0] * c.u + T.t[3] * c.v + T.t[6] * c.w;
    REAL ve = T.t[1] * c.u + T.t[4] * c.v + T.t[7] * c.w;
    REAL vd = T.t[2] * c.u + T.t[5] * c.v + T.t[8] * c.w;
    o[0] = c.pe; o[1] = c.pn; o[2] = -c.pd; o[3] = phi; o[4] = theta; o[5] = psi;
    o[6] = ve; o[7] = vn; o[8] = -vd;
}

#if MASS_TANKS
// FGMassBalance::GetPointmassInertia 한 점의 기여를 누적한다
__device__ __forceinline__ void tk_pm(REAL m, REAL rx, REAL ry, REAL rz, REAL cx, REAL cy, REAL cz,
                                      REAL& ixx, REAL& iyy, REAL& izz, REAL& ixz) {
    REAL vx = (rx - cx) * TK_AX, vy = (ry - cy) * TK_AY, vz = (rz - cz) * TK_AZ;
    ixx = ixx + m * (vy * vy + vz * vz);
    iyy = iyy + m * (vx * vx + vz * vz);
    izz = izz + m * (vx * vx + vy * vy);
    ixz = ixz + (-m * vx * vz);
}
#endif

// ================================================================ 프레임 앞부분
// 질량(직전 프레임까지 태운 연료) + FGAuxiliary + 조종석 하중배수
__device__ __forceinline__ void core_pre(Core& c, Pre& s, const Env& E) {
    const REAL one = (REAL)1;
#if MASS_TANKS
    {
        REAL w = TK_WFIX + (((c.f0 + c.f1) + c.f2) + c.f3);
        REAL mx = TK_MOM[0] + (((c.f0 * TK_R[0] + c.f1 * TK_R[3]) + c.f2 * TK_R[6]) + c.f3 * TK_R[9]);
        REAL my = TK_MOM[1] + (((c.f0 * TK_R[1] + c.f1 * TK_R[4]) + c.f2 * TK_R[7]) + c.f3 * TK_R[10]);
        REAL mz = TK_MOM[2] + (((c.f0 * TK_R[2] + c.f1 * TK_R[5]) + c.f2 * TK_R[8]) + c.f3 * TK_R[11]);
        REAL cx = mx / w, cy = my / w, cz = mz / w;
        REAL ixx = TK_JXX, iyy = TK_JYY, izz = TK_JZZ, ixz = TK_JXZ;
        tk_pm(TK_ME, TK_CGE_X, TK_CGE_Y, TK_CGE_Z, cx, cy, cz, ixx, iyy, izz, ixz);
        tk_pm(TK_MP, TK_PR_X, TK_PR_Y, TK_PR_Z, cx, cy, cz, ixx, iyy, izz, ixz);
        tk_pm(c.f0 / G_STD, TK_R[0], TK_R[1], TK_R[2], c.cgtx, c.cgty, c.cgtz, ixx, iyy, izz, ixz);
        tk_pm(c.f1 / G_STD, TK_R[3], TK_R[4], TK_R[5], c.cgtx, c.cgty, c.cgtz, ixx, iyy, izz, ixz);
        tk_pm(c.f2 / G_STD, TK_R[6], TK_R[7], TK_R[8], c.cgtx, c.cgty, c.cgtz, ixx, iyy, izz, ixz);
        tk_pm(c.f3 / G_STD, TK_R[9], TK_R[10], TK_R[11], c.cgtx, c.cgty, c.cgtz, ixx, iyy, izz, ixz);
        c.cgtx = cx; c.cgty = cy; c.cgtz = cz;          // 다음 프레임 탱크 항의 기준
        s.mass_slugs = w / G_STD;
        s.jxx = ixx; s.jyy = iyy; s.jzz = izz; s.jxz = ixz;
        s.cgx = cx; s.cgy = cy; s.cgz = cz;
        s.fuel_tot = ((c.f0 + c.f1) + c.f2) + c.f3;
    }
#else
    {
        REAL ff = cl(c.fuel, MS_F[0], MS_F[MS_N - 1]);
        int k1 = 0;
        while (k1 < MS_N && MS_F[k1] < ff) ++k1;       // bucketize(right=False)
        if (k1 < 1) k1 = 1;
        if (k1 > MS_N - 1) k1 = MS_N - 1;
        REAL f0 = MS_F[k1 - 1], f1 = MS_F[k1];
        REAL mw = cl((ff - f0) / (f1 - f0), (REAL)0, one);
        REAL mrow[7];
        #pragma unroll
        for (int k = 0; k < 7; ++k) mrow[k] = MS_T[7 * (k1 - 1) + k] * (one - mw) + MS_T[7 * k1 + k] * mw;
        s.mass_slugs = (EMPTY_LBS + c.fuel) / G_STD;
        s.jxx = mrow[0]; s.jyy = mrow[1]; s.jzz = mrow[2]; s.jxz = mrow[3];
        s.cgx = mrow[4]; s.cgy = mrow[5]; s.cgz = mrow[6];
        s.fuel_tot = c.fuel;
    }
#endif
    s.m_kg = s.mass_slugs * SLUG;
    s.Ixx = s.jxx * SLUGFT2; s.Iyy = s.jyy * SLUGFT2; s.Izz = s.jzz * SLUGFT2; s.Ixz = s.jxz * SLUGFT2;

    euler(c.qw, c.qx, c.qy, c.qz, s.phi, s.theta, s.psi);
    s.h_m = -c.pd;
    s.h_ft = s.h_m / FT;
    s.vt_ms = cmin(SQRT(c.u * c.u + c.v * c.v + c.w * c.w), (REAL)1e-6);
    s.vt_fps = s.vt_ms / FT;
    s.alpha = ATAN2(c.w, c.u);
    s.beta = ATAN2(c.v, cmin(SQRT(c.u * c.u + c.w * c.w), (REAL)1e-9));
    REAL aT, P_psf, a_fps;
    atmos(s.h_ft, aT, P_psf, s.rho, a_fps);
    s.mach = s.vt_fps / a_fps;
    s.qbar = (REAL)0.5 * s.rho * s.vt_fps * s.vt_fps;
    s.vc_kts = vcal_kts(s.mach, P_psf);
    s.T = dcm_l2b(c.qw, c.qx, c.qy, c.qz);
    const REAL* T = s.T.t;
    s.vn = T[0] * c.u + T[3] * c.v + T[6] * c.w;
    s.ve = T[1] * c.u + T[4] * c.v + T[7] * c.w;
    s.vd = T[2] * c.u + T[5] * c.v + T[8] * c.w;
    s.vg_fps = HYPOT(s.vn, s.ve) / FT;
    s.h_dot_ms = -s.vd;
    // 관성계 각속도 = pqr + Tl2b w_earth (w_earth = (WE_N, 0, WE_D))
    s.web_x = T[0] * E.WE_N + T[2] * E.WE_D;
    s.web_y = T[3] * E.WE_N + T[5] * E.WE_D;
    s.web_z = T[6] * E.WE_N + T[8] * E.WE_D;
    s.wix = c.p + s.web_x; s.wiy = c.q + s.web_y; s.wiz = c.r + s.web_z;
    // 조종석 하중배수 (a_body, wdot 는 직전 프레임 것)
    REAL ex = (s.cgx - EYE_X) / (REAL)12, ey = (EYE_Y - s.cgy) / (REAL)12, ez = (s.cgz - EYE_Z) / (REAL)12;
    REAL c1x = c.wdy * ez - c.wdz * ey, c1y = c.wdz * ex - c.wdx * ez, c1z = c.wdx * ey - c.wdy * ex;
    REAL c2x = s.wiy * ez - s.wiz * ey, c2y = s.wiz * ex - s.wix * ez, c2z = s.wix * ey - s.wiy * ex;
    REAL c3x = s.wiy * c2z - s.wiz * c2y, c3y = s.wiz * c2x - s.wix * c2z, c3z = s.wix * c2y - s.wiy * c2x;
    c.npx = (c.abx + (c1x + c3x)) / G_STD;
    c.npy_ = (c.aby + (c1y + c3y)) / G_STD;
    c.npz_ = (c.abz + (c1z + c3z)) / G_STD;
}

// ================================================================ 프레임 뒷부분
// 조종간(위층 규약 "+ = 오른쪽 / 당김") -> FLCS -> 터빈 -> 공력 -> 적분 -> 연료.
// `snap` 이면 이 프레임에 엔진을 트림 N2 로 스냅한다 (JSBSim tpTrim 한 프레임).
__device__ __forceinline__ void core_post(Core& c, const Pre& s, const Env& E,
                                          REAL ail_u, REAL ele_u, REAL rud_u, REAL thr_u,
                                          bool snap, REAL* d) {
    const REAL one = (REAL)1;
    const REAL p = c.p, q = c.q, r = c.r;
    // ================================================ FLCS
    REAL ail_c = cl(ail_u, -one, one);
    REAL ele_c = -cl(ele_u, -one, one);
    REAL rud_c = -cl(rud_u, -one, one);
    REAL thr_c = cl(thr_u, (REAL)0, one);
    // 1. 플랩
    REAL tef_pos_rad = (c.d_vc < TEF_VC) ? TEF_LO : ((c.d_mach > MACH_HI) ? TEF_HI_MACH : (REAL)0);
    REAL tef_target = cl(tef_pos_rad * TEF_NORM_GAIN, -one, one);
    {
        REAL pos_pos = cmin(c.tef, (REAL)0);
        REAL tgt_pos = cmin(tef_target, (REAL)0);
        REAL moved = pos_pos + cl(tgt_pos - pos_pos, -TEF_RATE, TEF_RATE);
        c.tef = (tef_target < (REAL)0 && moved <= (REAL)0) ? tef_target : moved;
    }
    // 2. 롤
    REAL e_roll = ail_c - ROLL_RATE_GAIN * c.d_p;
    REAL roll_pid = (ROLL_KP * e_roll + c.i_roll) + ROLL_KD * ((e_roll - c.e_roll_p) * INV_DT);
    c.e_roll_p = e_roll;
    REAL rrc = cl(roll_pid + ail_c, -one, one);
    REAL aileron_pos_rad = rrc * AIL_RANGE;
    c.ail_pn = c.ail_pn + cl(rrc - c.ail_pn, -AIL_RATE, AIL_RATE);
    REAL ail_sc = c.ail_pn * (one - (REAL)0.85 * cl(c.d_mach, (REAL)0, one));
    REAL lflap = cl(-c.tef - ail_sc, -one, one);
    REAL rflap = cl(c.tef - ail_sc, -one, one);
    REAL flaperon_mix = (lflap + rflap) * FLAP_MIX;
    // 3. 피치
    REAL nz_corr = COS(s.theta) * COS(s.phi);
    REAL g_load_c = c.d_nz - nz_corr;
    REAL elev_lim = cl(ele_c + c.ptrim, ELEV_MIN, ELEV_MAX);
    REAL aa = cmax(ABS(c.d_alpha), SCHED_A2);
    REAL auth = (aa <= SCHED_A1) ? one - SCHED_SLOPE1 * aa : SCHED_Y1 * (SCHED_A2 - aa) * SCHED_INV;
    REAL elev_sched = elev_lim * auth;
    REAL alpha_lim = ALPHA_LIM_GAIN * c.d_alpha;
    REAL e_pitch = (elev_sched + PITCH_RATE_GAIN * c.d_q) - G_LOAD_GAIN * g_load_c;
    REAL g_pid = cl((PITCH_KP * e_pitch + c.i_pitch) + PITCH_KD * ((e_pitch - c.e_pitch_p) * INV_DT), -one, one);
    c.e_pitch_p = e_pitch;
    REAL pitch_sched = cl((elev_sched + alpha_lim) + g_pid, -one, one);
    c.ele_pn = c.ele_pn + cl(pitch_sched - c.ele_pn, -ELE_RATE, ELE_RATE);
    REAL elevator_pos_rad = c.ele_pn * ELE_RANGE;
    // 4. 요
    REAL vgg = cl((c.d_vg - (REAL)80) * VG_INV20, (REAL)0, one) * (REAL)15
             + cl((c.d_vg - (REAL)100) * VG_INV50, (REAL)0, one) * (REAL)85;
    REAL e_yaw = (rud_c + c.d_r * vgg) + YAW_LOAD_GAIN * c.d_ny;
    REAL y_pid = cl((YAW_KP * e_yaw + c.i_yaw) + YAW_KD * ((e_yaw - c.e_yaw_p) * INV_DT), -one, one);
    c.e_yaw_p = e_yaw;
    REAL yaw_sched = cl((rud_c + (REAL)0) + y_pid, -one, one);
    REAL rud_pn = y_pid + cl(yaw_sched - y_pid, -RUD_RATE, RUD_RATE);
    REAL rudder_pos_rad = rud_pn * RUD_RANGE;
    // 6. 앞전 플랩 (기어 내림, WOW 0)
    REAL lef = (c.d_mach > MACH_HI) ? LEF_HI_MACH : (REAL)0;
    if (c.d_alpha > LEF_ALPHA_LO) lef = LEF_LO;
    // 지연 버퍼 -- 다음 프레임이 이번 값을 본다
    c.d_alpha = s.alpha; c.d_p = p; c.d_q = q; c.d_r = r; c.d_nz = c.npz_; c.d_ny = c.npy_;
    c.d_mach = s.mach; c.d_vc = s.vc_kts; c.d_vg = s.vg_fps;

    // ================================================ 추진
    REAL tpos = (REAL)2 * thr_c;
    REAL aug = cl(tpos - one, (REAL)0, one);
    REAL pos_cmd = cl(tpos, (REAL)0, one);
    REAL thrust;
    {
        REAL density_ratio = s.rho / RHO_SL_TURB;
        REAL nn = cmax(c.n2norm + (REAL)0.1, one);
        REAL om = one - nn;
        REAL denom = one + (REAL)3 * (om * om * om) + (one - density_ratio);
        REAL up = TB_UP / denom, dn = TB_DN / denom;
        REAL target = TB_IDLE + pos_cmd * TB_FACTOR;
        bool rise = c.n2 < target;
        REAL stp = rise ? DT * up : -DT * dn;
        c.n2 = rise ? tmin(c.n2 + stp, target) : tmax(c.n2 + stp, target);
        c.n2norm = (c.n2 - TB_IDLE) / TB_FACTOR;
        thrust = turb_thrust(c.n2norm, aug, s.mach, s.h_ft);
    }
    if (snap) {
        c.n2 = TB_IDLE + pos_cmd * TB_FACTOR;
        c.n2norm = pos_cmd;
        thrust = turb_thrust(pos_cmd, aug, s.mach, s.h_ft);
        c.ff_pps = fuel_flow(thrust, aug, c.n2norm);
    }

    // ================================================ 공력
    REAL X_qbar = s.qbar, X_alpha = s.alpha, X_beta = s.beta, X_mach = s.mach;
    REAL X_p = p, X_q = q, X_r = r;
    REAL X_aileron = aileron_pos_rad, X_elevator = elevator_pos_rad, X_rudder = rudder_pos_rad;
    REAL X_lef = lef, X_speedbrake = (REAL)0, X_flaperon_mix = flaperon_mix, X_gear = one;
    REAL X_h_b_mac = (REAL)100;
    REAL vtc = cmin(s.vt_fps, (REAL)1e-6);
    REAL X_bi2vel = (one / ((REAL)2 * vtc)) * AERO_BW;
    REAL X_ci2vel = (one / ((REAL)2 * vtc)) * AERO_CBAR;
    REAL X_kCLge;
    { int ki; REAL kt; bracket(KGE_BP, KGE_N, X_h_b_mac, ki, kt);
      X_kCLge = lerp_t(KGE_D[ki], KGE_D[ki + 1], kt); }
@@AERO_CODE@@
    REAL ca = COS(s.alpha), sa = SIN(s.alpha), cb = COS(s.beta), sb = SIN(s.beta);
    REAL nx_ = -aD, ny_ = aY, nz_ = -aL;
    REAL fbx_a = ca * cb * nx_ - ca * sb * ny_ - sa * nz_;
    REAL fby_a = sb * nx_ + cb * ny_;
    REAL fbz_a = sa * cb * nx_ - sa * sb * ny_ + ca * nz_;
    // 무게중심 둘레로
    REAL dx = (s.cgx - RP_X) / (REAL)12, dy = (RP_Y - s.cgy) / (REAL)12, dz = (s.cgz - RP_Z) / (REAL)12;
    REAL l_cg = al + dy * fbz_a - dz * fby_a;
    REAL m_cg = am + dz * fbx_a - dx * fbz_a;
    REAL n_cg = an + dx * fby_a - dy * fbx_a;
    // 합력
    REAL fbx = fbx_a + thrust;
    REAL Fx = fbx * LBF, Fy = fby_a * LBF, Fz = fbz_a * LBF;
    REAL Mx = l_cg * LBFFT, My = (m_cg + (s.cgz / (REAL)12) * thrust) * LBFFT, Mz = n_cg * LBFFT;
    // 관성계 각가속도 wdot_i = J^-1 (M - w_i x J w_i)
    REAL jwx = s.Ixx * s.wix + s.Ixz * s.wiz, jwy = s.Iyy * s.wiy, jwz = s.Ixz * s.wix + s.Izz * s.wiz;
    REAL gmx = Mx - (s.wiy * jwz - s.wiz * jwy);
    REAL gmy = My - (s.wiz * jwx - s.wix * jwz);
    REAL gmz = Mz - (s.wix * jwy - s.wiy * jwx);
    REAL det = s.Ixx * s.Izz - s.Ixz * s.Ixz;
    c.wdx = (s.Izz * gmx - s.Ixz * gmz) / det;
    c.wdy = gmy / s.Iyy;
    c.wdz = (s.Ixx * gmz - s.Ixz * gmx) / det;
    c.abx = fbx / s.mass_slugs; c.aby = fby_a / s.mass_slugs; c.abz = fbz_a / s.mass_slugs;

    // ================================================ 6-DOF (rbdyn.derivatives + jsbsim 적분)
    const REAL u = c.u, v = c.v, w = c.w;
    const REAL* T = s.T.t;
    REAL gscale = RH_REF / (R_EARTH + s.h_m);
    gscale = gscale * gscale;
    REAL gz = E.G0 * gscale + E.ACENT_D * (R_EARTH - c.pd);
    REAL gbx = T[2] * gz, gby = T[5] * gz, gbz = T[8] * gz;
    REAL wcx = s.wix + s.web_x, wcy = s.wiy + s.web_y, wcz = s.wiz + s.web_z;
    REAL udot = (Fx / s.m_kg + gbx) - (wcy * w - wcz * v);
    REAL vdot = (Fy / s.m_kg + gby) - (wcz * u - wcx * w);
    REAL wdot = (Fz / s.m_kg + gbz) - (wcx * v - wcy * u);
    REAL pdot = c.wdx + (q * s.web_z - r * s.web_y);
    REAL qdot = c.wdy + (r * s.web_x - p * s.web_z);
    REAL rdot = c.wdz + (p * s.web_y - q * s.web_x);
    REAL qdw = (REAL)0.5 * (-(c.qx * p + c.qy * q + c.qz * r));
    REAL qdx = (REAL)0.5 * (c.qw * p + c.qy * r - c.qz * q);
    REAL qdy = (REAL)0.5 * (c.qw * q + c.qz * p - c.qx * r);
    REAL qdz = (REAL)0.5 * (c.qw * r + c.qx * q - c.qy * p);
    // 속도는 **국소 NED 에서** 적분한다 (JSBSim 은 관성계 -- rbdyn 머리말 1).
    // a_ned = Tb2l (uvw_dot + pqr x uvw) : 지구 기준 속도의 NED 미분
    REAL abx = udot + (q * w - r * v), aby = vdot + (r * u - p * w), abz = wdot + (p * v - q * u);
    REAL anN = T[0] * abx + T[3] * aby + T[6] * abz;
    REAL anE = T[1] * abx + T[4] * aby + T[7] * abz;
    REAL anD = T[2] * abx + T[5] * aby + T[8] * abz;
    // AB 이력 (방금 리셋된 판은 세 칸 모두 현재 값).  hu* 는 NED 속도 미분이다.
    c.hu2x = c.hu1x; c.hu2y = c.hu1y; c.hu2z = c.hu1z; c.hu1x = c.hu0x; c.hu1y = c.hu0y; c.hu1z = c.hu0z;
    c.hu0x = anN; c.hu0y = anE; c.hu0z = anD;
    c.hp2x = c.hp1x; c.hp2y = c.hp1y; c.hp2z = c.hp1z; c.hp1x = c.hp0x; c.hp1y = c.hp0y; c.hp1z = c.hp0z;
    c.hp0x = s.vn; c.hp0y = s.ve; c.hp0z = s.vd;
    if (c.fr) {
        c.hu1x = c.hu2x = anN; c.hu1y = c.hu2y = anE; c.hu1z = c.hu2z = anD;
        c.hp1x = c.hp2x = s.vn; c.hp1y = c.hp2y = s.ve; c.hp1z = c.hp2z = s.vd;
    }
    if (d) {
        d[0] = s.alpha; d[1] = s.beta; d[2] = s.mach; d[3] = s.qbar; d[4] = s.vc_kts; d[5] = s.vg_fps;
        d[6] = c.npz_; d[7] = c.npy_;
        d[8] = elevator_pos_rad; d[9] = aileron_pos_rad; d[10] = rudder_pos_rad; d[11] = lef;
        d[12] = (REAL)0; d[13] = flaperon_mix;
        d[14] = thrust; d[15] = c.n2; d[16] = s.fuel_tot; d[17] = s.mass_slugs;
        d[18] = s.jxx; d[19] = s.jyy; d[20] = s.jzz; d[21] = s.jxz; d[22] = s.cgx; d[23] = s.cgz;
        d[24] = fbx_a; d[25] = fby_a; d[26] = fbz_a; d[27] = l_cg; d[28] = m_cg; d[29] = n_cg;
        d[30] = fbx; d[31] = fby_a; d[32] = fbz_a; d[33] = Mx / LBFFT; d[34] = My / LBFFT; d[35] = Mz / LBFFT;
        d[36] = udot / FT; d[37] = vdot / FT; d[38] = wdot / FT; d[39] = pdot; d[40] = qdot; d[41] = rdot;
    }
    c.p = p + DT * pdot; c.q = q + DT * qdot; c.r = r + DT * rdot;
    c.qw = c.qw + DT * qdw; c.qx = c.qx + DT * qdx; c.qy = c.qy + DT * qdy; c.qz = c.qz + DT * qdz;
    {
        REAL nq = cmin(SQRT(c.qw * c.qw + c.qx * c.qx + c.qy * c.qy + c.qz * c.qz), (REAL)1e-12);
        c.qw = c.qw / nq; c.qx = c.qx / nq; c.qy = c.qy / nq; c.qz = c.qz / nq;
    }
    {
        // NED 속도를 AB2 로 밀고 **적분한 뒤의** 자세로 동체축에 되돌린다.
        REAL vn1 = s.vn + DT * ((REAL)1.5 * c.hu0x - (REAL)0.5 * c.hu1x);
        REAL ve1 = s.ve + DT * ((REAL)1.5 * c.hu0y - (REAL)0.5 * c.hu1y);
        REAL vd1 = s.vd + DT * ((REAL)1.5 * c.hu0z - (REAL)0.5 * c.hu1z);
        Dcm Tn = dcm_l2b(c.qw, c.qx, c.qy, c.qz);
        c.u = Tn.t[0] * vn1 + Tn.t[1] * ve1 + Tn.t[2] * vd1;
        c.v = Tn.t[3] * vn1 + Tn.t[4] * ve1 + Tn.t[5] * vd1;
        c.w = Tn.t[6] * vn1 + Tn.t[7] * ve1 + Tn.t[8] * vd1;
    }
    c.pn = c.pn + DT * ((AB3_0 * c.hp0x - AB3_1 * c.hp1x) + AB3_2 * c.hp2x);
    c.pe = c.pe + DT * ((AB3_0 * c.hp0y - AB3_1 * c.hp1y) + AB3_2 * c.hp2y);
    c.pd = c.pd + DT * ((AB3_0 * c.hp0z - AB3_1 * c.hp1z) + AB3_2 * c.hp2z);
    c.fr = false;
    // 연료 (FGTurbine 소모 -> 질량)
    c.ff_pps = cl(fuel_flow(thrust, aug, c.n2norm), c.ff_pps - FF_DN, c.ff_pps + FF_UP);
#if MASS_TANKS
    {
        REAL need = c.ff_pps * DT;
        int nh = (c.f0 > (REAL)0) + (c.f1 > (REAL)0) + (c.f2 > (REAL)0) + (c.f3 > (REAL)0);
        REAL per = need / (REAL)(nh > 0 ? nh : 1);
        REAL* fs[4] = {&c.f0, &c.f1, &c.f2, &c.f3};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            REAL fk = *fs[k];
            if (fk > (REAL)0) { REAL left = fk - per; *fs[k] = (left >= (REAL)0) ? left : (REAL)0; }
        }
#if TK_REFUEL
        int no = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) no += (((REAL)100 * *fs[k] / TK_CAP[k]) < (REAL)99.99);
        REAL add = TK_REFUEL_STEP / (REAL)(no > 0 ? no : 1);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            bool open_ = ((REAL)100 * *fs[k] / TK_CAP[k]) < (REAL)99.99;
            *fs[k] = tmin(*fs[k] + (open_ ? add : (REAL)0), TK_CAP[k]);
        }
#endif
    }
#else
    c.fuel = cl(c.fuel + DT * (REFUEL_PPS - c.ff_pps), (REAL)0, FUEL_CAP);
#endif
    if (d) {
        REAL ph2, th2, ps2;
        euler(c.qw, c.qx, c.qy, c.qz, ph2, th2, ps2);
        Dcm T2 = dcm_l2b(c.qw, c.qx, c.qy, c.qz);
        d[42] = c.u / FT; d[43] = c.v / FT; d[44] = c.w / FT; d[45] = c.p; d[46] = c.q; d[47] = c.r;
        d[48] = ph2; d[49] = th2; d[50] = ps2; d[51] = -c.pd / FT;
        d[52] = (T2.t[0] * c.u + T2.t[3] * c.v + T2.t[6] * c.w) / FT;
        d[53] = (T2.t[1] * c.u + T2.t[4] * c.v + T2.t[7] * c.w) / FT;
        d[54] = (T2.t[2] * c.u + T2.t[5] * c.v + T2.t[8] * c.w) / FT;
    }
}

#define CORE_ARGS \
    REAL* __restrict__ pos, REAL* __restrict__ uvw, REAL* __restrict__ quat, \
    REAL* __restrict__ pqr, unsigned char* __restrict__ fresh, \
    REAL* __restrict__ hu, REAL* __restrict__ hp, REAL* __restrict__ fst, \
    REAL* __restrict__ n2_b, REAL* __restrict__ n2n_b, REAL* __restrict__ fuel_b, \
    REAL* __restrict__ ff_b, const REAL* __restrict__ ptrim_b, \
    REAL* __restrict__ ab_b, REAL* __restrict__ wd_b, REAL* __restrict__ np_b, \
    REAL* __restrict__ cgt_b, const REAL* __restrict__ env
#define CORE_PTR CorePtr P = {pos, uvw, quat, pqr, fresh, hu, hp, fst, n2_b, n2n_b, fuel_b, ff_b, \
                              ptrim_b, ab_b, wd_b, np_b, cgt_b}; \
                 Env E = {env[0], env[1], env[2], env[3]};
"""

# 조종간 드라이버 -- `F16Stick.step(stick, substeps)` 와 같은 일
_STICK_KERNEL = r"""
extern "C" __global__ void __launch_bounds__(128)
stick_step(int N, int n_sub, CORE_ARGS,
           REAL* __restrict__ held_b, unsigned char* __restrict__ etrim_b,
           unsigned char* __restrict__ tframe_b, const REAL* __restrict__ stick,
           REAL* __restrict__ dbg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORE_PTR
    Core c;
    core_load(c, P, i, N);
    REAL h0 = held_b[4 * i], h1 = held_b[4 * i + 1], h2 = held_b[4 * i + 2], h3 = held_b[4 * i + 3];
    const REAL s0 = stick[4 * i], s1 = stick[4 * i + 1], s2 = stick[4 * i + 2], s3 = stick[4 * i + 3];
    bool etrim = etrim_b[i] != 0, tframe = tframe_b[i] != 0;
    #pragma unroll 1
    for (int k = 0; k < n_sub; ++k) {
        Pre s;
        core_pre(c, s, E);
        // 트림 뒤 첫 "진짜" 프레임(리셋 뒤 두 번째)은 N2 를 목표로 스냅한다
        bool snap = etrim && !tframe;
        core_post(c, s, E, h0, h1, h2, h3, snap, dbg ? dbg + ((size_t)i * n_sub + k) * NDBG : 0);
        etrim = etrim && !snap;
        tframe = false;
        h0 = s0; h1 = s1; h2 = s2; h3 = s3;          // JSBSim 이 이제 들고 있는 스틱
    }
    core_store(c, P, i, N);
    held_b[4 * i] = s0; held_b[4 * i + 1] = s1; held_b[4 * i + 2] = s2; held_b[4 * i + 3] = s3;
    etrim_b[i] = etrim ? 1 : 0;
    tframe_b[i] = tframe ? 1 : 0;
}
"""


def core_source(aero, turb, mass, dbl: bool) -> str:
    """코어 장치 코드 (커널 없음).  드라이버가 뒤에 자기 커널을 붙인다."""
    head = "".join(f"#define {k} {v}\n" for k, v in _MATH[dbl].items())
    head += f"#define NDBG {NDBG}\n"
    head += _core_consts(dbl) + _tables(aero, turb, dbl) + _mass_tables(mass, dbl)
    decl, code = _aero_code(aero, dbl)
    return head + _CORE.replace("@@AERO_DECL@@", decl).replace("@@AERO_CODE@@", code)


def env_tensor(rb, device, dtype) -> torch.Tensor:
    """`RigidBody6DOF` 의 위도·중력 상수 (G0, w_earth N, w_earth D, 원심 D)."""
    return torch.stack((torch.tensor(rb.gravity, dtype=torch.float64),
                        rb.w_earth_ned[0].double().cpu(), rb.w_earth_ned[2].double().cpu(),
                        rb.a_cent_ned[2].double().cpu())).to(device=device, dtype=dtype)


def check_plant(dyn) -> None:
    """커널이 옮긴 구성인지.  아니면 조용히 다른 물리를 돌리므로 여기서 멈춘다."""
    rb = dyn.rb
    if dyn.device.type != "cuda":
        raise RuntimeError("fused_core 는 cuda 전용이다")
    if rb.integrator != "jsbsim":
        raise ValueError("integrator='jsbsim' 만 옮겼다")
    if not (rb._earth_rotation and rb._frame_correction and rb._gravity_varies):
        raise ValueError("rbdyn 기본 구성(자전·프레임 보정·고도별 중력)만 옮겼다")
    if float(rb.w_earth_ned[1]) != 0.0 or float(rb.a_cent_ned[0]) != 0.0 or float(rb.a_cent_ned[1]) != 0.0:
        raise ValueError("지구자전 벡터의 동쪽 성분·원심 수평 성분은 옮기지 않았다")
    f = dyn.flcs
    if not f.fast_pid or f.fbw_override or f.gear_wow != 0.0 or f.gear_pos != 1.0:
        raise ValueError("FLCS 는 fast_pid, fbw 해제, 기어 내림(WOW 0) 구성만 옮겼다")
    m = dyn.mass
    if hasattr(m, "_TAB") and not m.enabled:
        raise ValueError("질량 표 모델은 급유 램프 켬(enabled) 만 옮겼다")


def core_state(dyn) -> tuple:
    """코어 커널 인자 순서의 상태 텐서."""
    rb = dyn.rb
    cgt = getattr(dyn, "_cg_tank", None)
    if cgt is None or dyn.fuel.ndim == 1:
        cgt = torch.zeros(0, device=dyn.device, dtype=dyn.dtype)
    return (rb.pos_ned, rb.uvw, rb.quat, rb.pqr, rb._fresh,
            rb._hist["uvw"], rb._hist["pos"], dyn.flcs._state,
            dyn.n2, dyn.n2norm, dyn.fuel, dyn.ff_pps, dyn.pitch_trim,
            dyn.a_body, dyn.wdot_i, dyn.n_pilot, cgt)


_KERNELS: dict = {}


#: `--fmad=false` -- 곱·합을 따로 반올림한다 (FMA 로 묶지 않는다).  torch 는 연산마다
#: 결과를 한 번씩 반올림하므로 이쪽이 torch 에 더 가깝다 (float32 에서 포화 분기가 torch 와
#: 다르게 뒤집히는 일이 줄어든다).  비용은 약 2 %.  코드 구조를 바꿔도 결과가 안 흔들린다는
#: 덤도 있다 (FMA 묶음은 컴파일러가 정한다).
NVRTC_OPTS = ["-std=c++17", "--fmad=false"]


def _load(key, make_src, name):
    from jsbsim_f16_cuda import nvrtc
    if key not in _KERNELS:
        _KERNELS[key] = nvrtc.load(make_src(), name, NVRTC_OPTS)
    return _KERNELS[key]


def _mass_key(mass) -> tuple:
    if hasattr(mass, "_TAB"):
        return ("table",)
    return ("tanks", bool(mass.refuel))


class FusedStick:
    """`f16_core.F16Stick` 에 붙는다.  상태 텐서는 **그쪽 것을 그대로** 쓴다."""

    BLOCK = 128

    def __init__(self, dyn) -> None:
        check_plant(dyn)
        self.dyn = dyn
        dbl = dyn.dtype == torch.float64
        key = ("stick", dbl, _mass_key(dyn.mass), torch.cuda.current_device())
        self.k = _load(key, lambda: core_source(dyn.aero, dyn.turbine, dyn.mass, dbl)
                       + _STICK_KERNEL, "stick_step")
        self.env = env_tensor(dyn.rb, dyn.device, dyn.dtype)
        self._null = torch.zeros(0, device=dyn.device, dtype=dyn.dtype)
        for t in self.state():
            if t.numel() and not t.is_contiguous():
                raise ValueError("fused_core: 상태 텐서가 연속이 아니다")

    def state(self) -> tuple:
        d = self.dyn
        return core_state(d) + (d.stick, d._engine_trim, d._trim_frame)

    def step(self, stick: torch.Tensor, substeps: int = 1, dbg=None) -> None:
        """`F16Stick.step` 과 같은 계약.  `stick` (N, 4)."""
        d = self.dyn
        s = stick.to(device=d.device, dtype=d.dtype).reshape(d.N, 4).contiguous()
        st = core_state(d)
        self.k(grid=((d.N + self.BLOCK - 1) // self.BLOCK, 1, 1), block=(self.BLOCK, 1, 1),
               args=[d.N, int(substeps), *st, self.env, d.stick, d._engine_trim,
                     d._trim_frame, s, dbg if dbg is not None else self._null])


def attach_stick(dyn) -> FusedStick:
    """`dyn.step` 을 커널로 바꿔 끼운다 (인스턴스 속성).  `detach(dyn)` 로 되돌린다."""
    f = FusedStick(dyn)
    dyn.step = f.step
    dyn._fused = f
    return f


def detach(dyn) -> None:
    if "step" in vars(dyn):
        del dyn.step
    dyn._fused = None
