# jsbsim-f16-cuda

**JSBSim 1.3.0 의 F-16 비행모델을 PyTorch 로 옮긴 배치 시뮬레이터.**  수만 대를
GPU 에서 한 번에 1/120 초씩 전진시킨다.  강화학습처럼 같은 비행모델을 아주 많이
돌려야 하는 곳을 위해 만들었다.

> *English summary.*  A batched PyTorch/CUDA port of the **F-16 model that ships with
> JSBSim 1.3.0** (`f16.xml` + F100-PW-229 engine + FLCS).  It steps tens of thousands of
> aircraft per call at the physics rate (1/120 s) from raw stick inputs, and is
> verified **frame by frame** against JSBSim itself: control-surface positions match
> exactly, body-axis accelerations to ~1e-5 ft/s², angular accelerations to ~1e-7 rad/s²
> (float64).  It is **not** a port of the JSBSim engine — other aircraft XML files will
> not run.  License: GPL-3.0-or-later (derived from `f16.xml`, GPL, and JSBSim, LGPL).

---

## 무엇이고 무엇이 아닌가

- **옮긴 것**: JSBSim 1.3.0 번들 `aircraft/f16/f16.xml` 의 비행제어(FLCS), 공력 계수표
  전부, `engine/F100-PW-229.xml` 터보팬(스풀·애프터버너·연료유량), 질량·관성
  (빈 기체 + 조종사 + 연료탱크 4 개), 표준대기, JSBSim 보조량(받음각·마하·교정대기속도·
  조종석 하중배수), 6 자유도 운동방정식과 JSBSim 과 같은 적분기(Adams-Bashforth),
  지구 자전(코리올리·원심)과 JSBSim 의 J2 중력 크기.
- **옮기지 않은 것**: JSBSim 엔진 자체(XML 해석기, 임의 기체).  **F-16 한 기종만** 된다.
  착륙·지면 접촉, 바람·난류, 연료 이송·투하, 연료 바닥 시 엔진 정지, 착륙장치 올림.
  자세한 것은 아래 [한계](#한계).
- **입력은 조종간 네 개**(aileron, elevator, rudder, throttle)다.  자동비행·유도
  로직은 들어 있지 않다.

## 설치

```bash
git clone <이 저장소> && cd jsbsim-f16-cuda
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 판 torch
pip install -e .                    # 또는 pip install -r requirements.txt
pip install jsbsim==1.3.0           # JSBSim 대조 도구를 돌릴 때만
```

필요한 것은 `torch`(2.1 이상), `numpy` 뿐이다.  C++·CUDA 툴킷 빌드는 없다.
`jsbsim` 은 검증 도구와 트림 표 재생성에만 쓴다.

## 빠른 시작

```python
import math, torch
from jsbsim_f16_cuda import F16Stick

N = 4096
dyn = F16Stick(N, device="cuda")                      # 위도 0, 기본 연료 3,000 lb

alt_m = torch.full((N,), 20000 * 0.3048, device="cuda")
pos_ned = torch.stack([torch.zeros(N, device="cuda"),   # 북 [m]
                       torch.zeros(N, device="cuda"),   # 동 [m]
                       -alt_m], -1)                     # 아래 [m] = -해면고도
psi = torch.rand(N, device="cuda") * 2 * math.pi       # 진방위 [rad]
vt = torch.full((N,), 450 * 0.514444, device="cuda")   # 진대기속도 [m/s]
ok = dyn.reset(pos_ned, psi, vt)                       # 수평 트림으로 시작

stick = torch.tensor([0.0, 0.3, 0.0, 0.9], device="cuda").expand(N, 4)
for _ in range(120 * 10):                              # 10 초
    dyn.step(stick)                                    # = JSBSim set_controls(); run()
s = dyn.state()                                        # pos_ned, euler, uvw, pqr, alpha, ...
```

`examples/quickstart.py` 는 4,096 대를 무작위 조종간으로 20 초 날리고,
`examples/bench.py` 는 처리량을 잰다.

## 처리량

`python examples/bench.py` (RTX 5070 Ti 16 GB, torch 2.11 + CUDA 12.8, float32).
한 번의 `step(stick, substeps=6)` 을 CUDA 그래프로 캡처해 재생한 값이다.

| 기체 수 | eager | CUDA 그래프 | 한 대가 실시간의 몇 배로 나나 (그래프) |
|---|---|---|---|
| 256 | 0.01 M | 0.18 M | 6 x |
| 1,024 | 0.03 M | 0.67 M | 5 x |
| 4,096 | 0.13 M | 2.70 M | 5 x |
| 16,384 | 0.54 M | 9.66 M | 5 x |
| 65,536 | 2.13 M | **24.9 M** | 3 x |

단위는 초당 기체·물리프레임.  CPU(같은 PC, float32)는 1,024 대에서 0.18 M.
측정 중 GPU 를 다른 작업과 나눠 쓰고 있었으므로 보수적인 값이다.

마지막 열은 기체 한 대당 초당 물리 프레임을 120 으로 나눈 것이다 (실시간이면 1 x).
CUDA 그래프 없이(eager) 부르면 파이썬·커널 실행 대기 때문에 수십 배 느리다 --
큰 배치에서는 꼭 그래프로 감싸라 ([CUDA 그래프](#cuda-그래프)).

## API

### `F16Stick(n, device="cuda", dtype=torch.float32, *, lat0_deg=0.0, refuel=False, fuel_lbs=None, gravity_ms2=None, integrator="jsbsim")`

F-16 `n` 대.  모든 상태가 `(n, ...)` 텐서다.

| 인자 | 뜻 |
|---|---|
| `lat0_deg` | 평면(접평면)의 기준 위도.  지구 자전 항과 J2 중력 크기가 여기서 정해진다 |
| `refuel` | JSBSim `propulsion/refuel` (100 lb/s 로 안 찬 탱크를 똑같이 나눠 채운다).  기본 꺼짐 = 연료 정상 소모 |
| `fuel_lbs` | 탱크 4 개의 초기 연료 [lb].  기본 f16.xml 값 (1,500, 1,500, 0, 0).  용량 (3,486, 3,486, 2,991, 2,991) |
| `gravity_ms2` | 20,000 ft 에서의 중력 크기를 직접 줄 때 (기본: 위도의 JSBSim J2 값) |
| `integrator` | `"jsbsim"`(JSBSim 기본 적분기 조합) / `"euler"` / `"rk4"` |

### `reset(pos_ned, psi, vt_ms, fuel_lbs=None, mask=None) -> ok`

**수평 트림 비행**으로 시작한다 (JSBSim `do_trim(1)` 이 만드는 상태).

| 인자 | 모양 | 단위 |
|---|---|---|
| `pos_ned` | `(n, 3)` | m.  북, 동, **아래** (아래 = -해면고도) |
| `psi` | `(n,)` | rad.  진방위 (북 0, 동 +π/2) |
| `vt_ms` | `(n,)` | m/s.  진대기속도 |
| `fuel_lbs` | `(n, 4)` | lb.  탱크별 연료 |
| `mask` | `(n,)` bool | 켜진 기체만 리셋 (그래프 안에서 쓸 수 있다) |

트림 자세(받음각·피치·뱅크), 피치 트림, 트림 스로틀, N2, FLCS 지연버퍼는 JSBSim 에서 잰
표(`f16_trim.npz`)를 삼선형 보간해 넣는다.  돌려주는 `ok` 는 그 점이 표 안이고 주변 격자가
전부 트림 가능했는지다 (저속·고고도처럼 수평비행이 안 되는 곳은 `False`).

### `step(stick, substeps=1)`

`stick` 은 `(n, 4)` = (aileron, elevator, rudder, throttle).

| 채널 | 범위 | 양수의 뜻 | JSBSim 프로퍼티 |
|---|---|---|---|
| aileron | [-1, 1] | 오른쪽 롤 | `fcs/aileron-cmd-norm` = aileron |
| elevator | [-1, 1] | 당김 (기수 올림) | `fcs/elevator-cmd-norm` = -elevator |
| rudder | [-1, 1] | 오른쪽 요 | `fcs/rudder-cmd-norm` = -rudder |
| throttle | [0, 1] | 1 = 애프터버너 최대 | `fcs/throttle-cmd-norm` = throttle |

**`step(u)` 한 번 = JSBSim 의 `set_controls(u); run()` 한 번.**  JSBSim 은 프레임을
적분(Propagate)으로 시작하므로, 지금 준 조종은 **다음 프레임의 적분**부터 힘이 된다.
이 플랜트도 똑같이 직전 조종을 들고 있다가 한 프레임 늦게 쓴다.  그래서 `step` 뒤의
위치·자세·속도·각속도는 JSBSim `run()` 직후와 같은 시점이다.
`substeps=k` 는 같은 조종으로 k 프레임을 연달아 민다.

### `state() -> dict`

| 키 | 모양 | 단위 |
|---|---|---|
| `pos_ned` | `(n, 3)` | m |
| `quat` | `(n, 4)` | local NED → body 쿼터니언 (w, x, y, z), JSBSim 과 같은 방향 |
| `euler` | `(n, 3)` | rad.  (phi 뱅크, theta 피치, psi 방위), 3-2-1 |
| `uvw` | `(n, 3)` | m/s.  body 속도 |
| `vel_ned` | `(n, 3)` | m/s |
| `pqr` | `(n, 3)` | rad/s.  body 각속도 |
| `alpha`, `beta` | `(n,)` | rad |
| `n2` | `(n,)` | % |
| `fuel_lbs` | `(n, 4)` | lb |
| `n_pilot` | `(n, 3)` | g.  조종석 하중배수, JSBSim `accelerations/n-pilot-*-norm` 과 같은 부호 (수평비행 z = -1) |

살아 있는 텐서가 필요하면 `dyn.rb.pos_ned` · `dyn.rb.uvw` · `dyn.rb.quat` · `dyn.rb.pqr`,
`dyn.flcs.*`, `dyn.n2`, `dyn.fuel` 을 직접 읽는다 (복사 없음, 제자리 갱신).

### CUDA 그래프

상태는 전부 **제자리로만** 갱신한다 (주소가 안 바뀐다).  그래서 `step` 을 통째로 캡처할
수 있다:

```python
stick_buf = torch.zeros(N, 4, device="cuda")
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        dyn.step(stick_buf, 6)                         # 몸풀기 (캡처 전 필수)
torch.cuda.current_stream().wait_stream(side)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    dyn.step(stick_buf, 6)
for _ in range(1000):
    stick_buf.copy_(my_policy(...))                    # 입력은 같은 버퍼에 써 넣는다
    g.replay()
```

`reset(..., mask=)` 도 호스트 동기화가 없어서 그래프 안에 넣을 수 있다.

## JSBSim 과 얼마나 같은가

전부 이 저장소의 도구로 다시 잴 수 있다 ([검증 직접 돌리기](#검증-직접-돌리기)).

### 한 프레임 대조 (`fdm_verify`)

JSBSim 을 여러 입력(계단·더블릿·무작위 등 13 가지)과 비행조건 4 가지(320 kt/5,000 ft ~
600 kt/30,000 ft)로 날리다가, 한가운데 상태를 **통째로** GPU 플랜트에 심고 한 프레임만
전진시켜 모든 중간량을 비교한다.  누적이 없어서 남는 차이는 모델 차이다
(방법: [`docs/frame_parity.md`](docs/frame_parity.md)).

| 양 | float64, 위도 0 | float64, 위도 60 | float32, 위도 0 | 단위 |
|---|---|---|---|---|
| u̇ (전진 가속도) | 7.2e-6 | 4.0e-4 | 1.1e-5 | ft/s² |
| v̇ | 2.1e-5 | 1.8e-5 | 2.3e-5 | ft/s² |
| ẇ | 1.4e-4 | 1.8e-4 | 1.5e-4 | ft/s² |
| ṗ (롤 각가속도) | 1.2e-7 | 1.2e-7 | 2.3e-6 | rad/s² |
| q̇ | 1.4e-8 | 1.4e-8 | 1.2e-6 | rad/s² |
| ṙ | 4.4e-9 | 4.5e-9 | 1.6e-7 | rad/s² |
| 조종면 (승강타·보조익·방향타) | **0** | **0** | 4e-8 | rad |
| 추력 | 3.4e-6 | 3.4e-6 | 2.9e-3 | lbf (추력 수천~2만 lbf) |
| N2 | 1.6e-9 | 1.6e-9 | 6.7e-6 | % |

값은 52 조합(13 입력 × 4 조건) 중 최대 |오차| 다.  float32 열은 식의 차이가 아니라
반올림(eps 1.2e-7)이다.

### 질량 (`f16_check mass`)

탱크 네 개를 점질량으로 JSBSim 과 같은 식·같은 순서로 계산한다 (표 보간이 아니다).

| 시험 | 무게 | 무게중심 | 관성 (ixx, iyy, izz, ixz) | 탱크별 연료 |
|---|---|---|---|---|
| 무작위 탱크 배치 200 개 (정적) | 4e-12 lb | 3e-14 in | ≤ 3e-11 slug·ft² | -- |
| 20 초 무작위 스로틀, 급유 끔/켬, 만탱크, 바닥나는 탱크 (동적) | 4e-12 | 3e-14 | ≤ 3e-11 | **0** |

### 트림 표 (`f16_trim_build --check`)

`reset` 이 쓰는 트림 값은 JSBSim `do_trim(1)` 을 해면고도 0 ~ 45,000 ft (2,500 ft 간격) ×
150 ~ 800 kt (10 kt) × 연료 7 점(500 ~ 12,954 lb)에서 8,778 번 잰 표다 (트림 수렴 80.7 %).
격자 **사이** 무작위 200 점에서:

| 양 | 보간 대 JSBSim, 중앙 | 최대 |
|---|---|---|
| 트림 피치 | 5.7e-5 rad | 9.7e-4 rad |
| 피치 트림 명령 | 2.0e-4 | 1.2e-2 |
| 트림 스로틀 | 3.6e-4 | 7.2e-3 |
| N2 | 5.1e-3 % | 0.96 % |

그 상태에서 트림 조종을 그대로 들고 10 초 날리면 고도 변화가 GPU 중앙 2.0 ft, JSBSim 자신
1.8 ft 다 (JSBSim 트림도 완벽한 정지가 아니다).  격자점 위에서는 GPU 가 JSBSim 보다 오히려
덜 움직인다.  최악은 스케줄이 꺾이는 칸 사이였다 -- 32,276 ft / 465 kt 에서 10 초에
+38 ft (JSBSim +1.4 ft).

### 구성요소 (`validate_aero`)

JSBSim 에서 읽은 상태를 그대로 넣고 계수표만 대조한 것 (포락선 전체 2,000 표본):
공력 힘·모멘트 상대오차 중앙 6e-8 ~ 1.2e-7 (최대 1e-4 이내), 표준대기 5.7e-7, 추력표 5.7e-6,
스풀 계단응답 N2 최대 1e-4 %.

### 긴 궤적에 대해

**긴 궤적 오차로는 정합을 판정할 수 없다.**  JSBSim 을 자기 자신과 비교해도 초기속도가
1e-6 kt 다르면 수백 초 뒤 km 단위로 갈린다 (특히 무작위 조종에서).  이 포팅도 수십 초가
지나면 JSBSim 과 벌어진다 -- 모델이 달라서가 아니라 비행이 혼돈적이기 때문이다.  위
한 프레임 표가 모델 정합의 기준이다.

## 옮기면서 찾은 함정

조용히 틀리는(에러 없이 값만 다른) 자리들이다.  전부 한 프레임 대조로 드러났다.

- **코리올리 계수 2.**  JSBSim 은 `-(pqr + 2Ω) × uvw` 다.  한 번만 넣으면 v̇ 가 26 배 틀린다.
- **자전 원심가속도의 연직 몫**(g 의 0.2 %)은 넣고 수평 몫은 넣지 않는다 (측지 연직에 흡수됨).
- **연료유량의 변화율 제한** (`FGTurbine::Seek`, +5,000 / -10,000 pph/s).  스풀 중 40 배 차이.
- **질량은 직전 프레임 연료로**, **탱크 관성 항은 두 프레임 전 무게중심으로** 잰다 (JSBSim 모델 실행 순서).
- **조종 입력 한 프레임 지연.**  JSBSim 은 프레임을 적분으로 시작한다.
- **FLCS 는 자기 안에서 한 프레임 미룬다** -- 밖에서 또 미루면 고하중에서만 틀린다.
- **교정대기속도**: 충격압은 국소 정압 기준, 역산은 해면 정압 기준.  섞으면 20,000 ft 에서 250 kt 가까이 틀리고 플랩·이득 스케줄이 통째로 어긋난다.
- **뒷전 플랩 초기값은 스위치다**, 보간하면 안 된다 (문턱 칸에서 존재하지 않는 중간값이 나온다).
- **조종석 하중배수**는 입력마다 나이가 다르다 (직전 프레임 비력·각가속도, 이번 프레임 각속도).
- **스로틀 두 배**: FLCS 출력(`throttle-pos-norm`, 0~2)을 엔진에 넣으면 스로틀이 두 번 곱해진다.  엔진은 `throttle-cmd-norm` 을 받는다.
- **트림 뒤 첫 프레임은 N2 를 목표로 스냅**한다 (JSBSim tpTrim 단계).
- **관성곱 부호**: `inertia/ixz-slugs_ft2` 값을 그대로 쓴다 (XML 의 `negated_crossproduct_inertia` 에 속으면 1,700 배).
- **Adams-Bashforth 이력**은 `[0]` 이 직전 프레임이다.

JSBSim 쪽 함정(엔진이 꺼진 채 트림됨, `run_ic()` 가 FCS 적분기를 안 지움, 지심·측지 위도
혼동 등)은 [`jsbsim_f16_cuda/f16_reference.py`](jsbsim_f16_cuda/f16_reference.py) 머리말에 있다.

## 검증 직접 돌리기

저장소 루트에서 (`pip install jsbsim==1.3.0` 필요):

```bash
python -m jsbsim_f16_cuda.fdm_verify --check          # 한 프레임 대조 (float64, 위도 0)
python -m jsbsim_f16_cuda.fdm_verify --check --lat 60
python -m jsbsim_f16_cuda.fdm_verify --dtype float32  # 표만 (임계는 float64 기준)
python -m jsbsim_f16_cuda.f16_check mass              # 질량·관성·연료
python -m jsbsim_f16_cuda.f16_trim_build --check      # 트림 표 대 JSBSim do_trim
python -m jsbsim_f16_cuda.validate_aero               # 공력·대기·추력 계수표
python -m jsbsim_f16_cuda.rbdyn_validate              # 강체 적분기 (JSBSim 힘 주입)
python -m jsbsim_f16_cuda.f16_trim_build              # 트림 표 다시 만들기 (1 분 남짓)
python -m jsbsim_f16_cuda.parse_f16                   # 계수표 f16_tables.npz 다시 만들기
```

JSBSim 기준은 `f16_reference.JSBSimF16Ref` 하나다 (JSBSim 표준 설정: 트림 뒤 조종 유지,
연료 정상 소모, 급유 없음).

## 한계

- **F-16 한 기종**, JSBSim 1.3.0 번들 `f16.xml` 기준.  다른 JSBSim 버전의 f16.xml 과는 다를 수 있다.
- **착륙장치는 내린 채 고정**이다 (JSBSim 이 모델을 로드한 직후 상태.  항력의 절반 가까이가 기어다).  착륙·이륙·지면 접촉은 없다.
- **대기는 표준대기뿐**, 바람·난류·돌풍 없음.
- **연료**: 탱크 사이 이송·투하 없음.  연료가 다 떨어져도 엔진이 꺼지지 않는다 (JSBSim 은 끈다).
- **`reset` 은 수평 트림만** 만든다.  임의 자세로 시작하려면 `dyn.rb` 상태를 직접 쓰면 되지만 FLCS·엔진은 그 자세의 트림이 아니다.
- **트림 표 범위**: 해면고도 0 ~ 45,000 ft, 150 ~ 800 kt, 연료 500 ~ 12,954 lb (내부 탱크를 먼저 채운 배치 기준), 위도 0 에서 측정.  표 밖이나 트림이 안 잡히는 곳은 `ok=False` 이고 가장 가까운 칸 값을 쓴다.  외부 탱크만 채운 배치나 다른 위도에서는 초기 트림이 약간 어긋난다.  스케줄이 꺾이는 칸 사이에서는 10 초에 수십 ft 움직일 수 있다.
- **평평한 지구**: 기준 위도의 접평면에 자전 항만 넣었다.  수송률·지구 곡률은 뺐으므로 기준점에서 수십 km 넘게 멀어지면 JSBSim(구면)과 벌어진다.  위도가 높을수록 u̇ 잔차가 커진다 (위도 60 에서 4e-4 ft/s²).
- **float32** 는 반올림 수준(eps 1.2e-7) 차이가 있다 (위 표).
- **입력 한 프레임 지연**은 JSBSim 과 같게 한 것이다 -- 조종이 곧바로 힘이 되는 모델을 기대하면 한 프레임 다르다.

## 파일

```
jsbsim_f16_cuda/
  f16_core.py        F16Stick (플랜트), TankMass (질량), TrimGrid (트림 표), JSBSim 보조량·연료유량
  rbdyn.py           6 자유도 운동방정식 + 적분기 (JSBSim 적분기 조합, 지구 자전)
  flcs.py            f16.xml <flight_control> -- 조종간 -> 조종면 6 개
  aero.py            f16.xml <aerodynamics> 계수표 -> 힘·모멘트
  propulsion.py      F100-PW-229 터보팬 (스풀, 애프터버너, 추력표) + 표준대기
  f16_interp.py      표 보간 (JSBSim 과 같은 끝값 처리)
  f16_tables.npz     f16.xml · F100-PW-229.xml 에서 뽑은 계수표 (parse_f16.py 가 만든다)
  f16_trim.npz       JSBSim do_trim(1) 으로 잰 트림 표 (f16_trim_build.py 가 만든다)
  f16_reference.py   JSBSim 기준기 (jsbsim 패키지 직접 세팅)
  fdm_verify.py      한 프레임 대조
  f16_check.py       질량 대조
  f16_trim_build.py  트림 표 생성·대조
  validate_aero.py   계수표·대기·추력 대조
  rbdyn_validate.py  강체 적분기 대조
  parse_f16.py       f16.xml -> f16_tables.npz
examples/
  quickstart.py      4,096 대 무작위 비행
  bench.py           처리량 (eager 대 CUDA 그래프)
docs/
  frame_parity.md    한 프레임 대조 방법과 찾은 오류
```

## 라이선스와 출처

- 이 저장소: **GPL-3.0-or-later** ([LICENSE](LICENSE)).
- JSBSim (<https://github.com/JSBSim-Team/jsbsim>, LGPL-2.1): 운동방정식·적분기·질량·엔진·
  보조량의 계산 순서와 식을 옮겼다.
- F-16 모델 `aircraft/f16/f16.xml` (Erik Hofman, GPL) 과 `engine/F100-PW-229.xml`: 비행제어
  논리와 계수표를 옮겼다 (`f16_tables.npz` 는 이 파일들에서 뽑은 데이터다).  f16.xml 머리말대로
  이 모델은 공개 자료로 만든 것이며 실제 기체 제조사와 무관하다.
