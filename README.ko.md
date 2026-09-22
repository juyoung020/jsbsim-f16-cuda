# jsbsim-f16-cuda

**JSBSim 의 F-16 을 손으로 짠 CUDA 커널로: GPU 한 장에서 초당 30.9 억 기체·프레임, JSBSim 과
한 프레임씩 대조.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA via NVRTC](https://img.shields.io/badge/CUDA-NVRTC%2C%20no%20toolkit-76b900)
· [English](README.md)

JSBSim 1.3.0 에 들어 있는 F-16 모델(`f16.xml` 의 비행제어·공력, F100-PW-229 엔진, 연료탱크,
JSBSim 적분기와 지구 자전을 쓴 6 자유도 운동방정식)을 CUDA C++ 로 옮겼다.  스레드 하나가 기체
하나를 날리고, 커널 한 번이 모든 기체를 1/120 초 물리 프레임 여러 개만큼 전진시킨다.  커널
본체는 손으로 짰고, 상수와 계수표는 파이썬이 모델 데이터에서 생성해 붙인 뒤 NVRTC 로 실행 중에
컴파일한다 -- 빌드 단계가 없다.  강화학습·몬테카를로·최적화처럼 같은 비행모델을 수십억 번
돌려야 하는 곳에서, JSBSim 의 숫자를 잃지 않고 GPU 한 장이 CPU 클러스터를 대신하도록 만들었다.

## 시뮬레이터 처리량 (물리만)

RTX 5070 Ti, Ryzen 9 9950X (16 코어 / 32 스레드).  float32, 호출당 6 프레임, GPU 는 다른 작업
없음 ([조건과 전체 숫자](docs/benchmarks.md)):

- CUDA 커널 262,144 대에서 **초당 30.9 억 기체·프레임** -- GPU 한 장으로 **F-16 2,500 만 대를
  실시간으로** 날릴 수 있는 양이다.
- **JSBSim CPU 1 코어의 25,300 배, 하드웨어 스레드 32 개 전부의 1,300 배** (CUDA 커널 대
  JSBSim `run()`, 물리만).
- 한 프레임 대조에서 **조종면이 JSBSim 과 똑같다** (오차 0).

![처리량](docs/images/throughput.png)

| 시뮬레이터 | 초당 기체·프레임 | 기체·프레임 하나의 비용 | JSBSim 1 코어 대비 (물리만) |
|---|---|---|---|
| JSBSim 1.3.0, 프로세스 1 개 | 122 k | 8.2 µs | 1 배 |
| JSBSim 1.3.0, 프로세스 32 개 | 2.38 M | 0.42 µs | 20 배 |
| torch CPU 백엔드, 16,384 대, 스레드 16 | 0.94 M | 1.1 µs | 7.7 배 |
| torch GPU 백엔드 + CUDA 그래프, 262,144 대 | 39.9 M | 25 ns | 327 배 |
| **CUDA 커널, 262,144 대** | **3.09 G** | **0.32 ns** | **25,300 배** |

비용 = 1 / 처리량 (배치 전체에 나눠 낸 값).  20 Hz 결정 한 스텝(6 프레임)은 그 6 배다:
기체 한 대당 JSBSim 1 코어로 49 µs, 커널로 1.9 ns.

## 학습 속도에는 어떻게 나타나나

위 숫자는 시뮬레이터만의 것이다.  학습 전체가 얼마나 빨라지는지는 나머지 파이프라인에 달렸다.
정책 추론·관측·보상·학습 알고리즘에 드는 시간을 `T_rest` 라 하면

`속도 향상 = (T_sim_기존 + T_rest) / (T_sim_새것 + T_rest)` -- 지금 학습 시간 중 시뮬레이터가
차지하는 비율이 `p` 라면 향상은 최대 `1 / (1 − p)` 다.

예시 하나 (보장값이 아니다) -- 같은 PC 에서 같은 비공개 PPO 학습기(정책 256×2 MLP)를 처음엔
CPU 의 JSBSim 으로, 다음엔 GPU 의 이 시뮬레이터(torch 백엔드 + CUDA 그래프)로 돌렸다:

| 학습 파이프라인 | 초당 env-step (학습 전체) | CPU 파이프라인 대비 |
|---|---|---|
| JSBSim 1.3.0, 워커 프로세스 28 개 (SB3 `SubprocVecEnv`), 롤아웃 CPU · 갱신 GPU | 약 4,200 | 1 배 |
| GPU 시뮬레이터, 병렬 환경 65,472 개, 미니배치 65,536 (학습 한 바퀴 벤치) | 525,692 | 125 배 |
| 같은 것, 장기 학습 실측 (이터마다 신경망 갱신이 더 붙음) | 약 317,000 | 약 76 배 |

env-step 하나 = 두 기체 에피소드 하나의 결정 한 스텝 (기체 2 대 × 물리 6 프레임).  125 배는 물리만이
아니라 학습 전체 기준이고, 대부분은 GPU 시뮬레이터가 가능하게 한 것 -- 수만 개의 병렬 환경과 큰
미니배치 -- 에서 나왔다.  물리는 그때도 이미 스텝 비용의 작은 몫이었다.  이후 그 물리를 CUDA 커널로
바꾸자 (정책 1024×2 의 더 큰 파이프라인에서) 학습 한 이터가 약 19.0 초에서 약 15.6 초가 됐고, 물리는
그 이터의 약 0.2 % 가 됐다.  [자세히](docs/benchmarks.md#simulator-vs-end-to-end-training).

## 빠른 시작

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 판 torch 아무거나
git clone https://github.com/juyoung020/jsbsim-f16-cuda && cd jsbsim-f16-cuda
pip install -e .                        # 검증 도구를 돌리려면 jsbsim==1.3.0 도
```

```python
import math, torch
from jsbsim_f16_cuda import F16Stick, attach_stick

N = 65536
dyn = F16Stick(N, device="cuda")              # 위도 0, 연료 3,000 lb, JSBSim 기본값
attach_stick(dyn)                             # 이제 step() 이 CUDA 커널로 돈다
pos_ned = torch.zeros(N, 3, device="cuda"); pos_ned[:, 2] = -20000 * 0.3048   # m, 아래 = -고도
ok = dyn.reset(pos_ned, psi=torch.rand(N, device="cuda") * 2 * math.pi,
               vt_ms=torch.full((N,), 450 * 0.514444, device="cuda"))        # 수평 트림
stick = torch.tensor([0.0, 0.3, 0.0, 0.9], device="cuda").expand(N, 4)      # 보조익, 승강타, 방향타, 스로틀
for _ in range(200):
    dyn.step(stick, substeps=6)               # = JSBSim set_controls(); run() 6 번
state = dyn.state()                           # pos_ned, euler, uvw, pqr, alpha, beta, fuel, ...
```

처음 `attach_stick` 때 커널을 컴파일하고(수 초) 디스크에 캐시한다.  입력은 조종간 그대로다
(보조익·승강타·방향타 [−1, 1], 스로틀 [0, 1]).  자동비행 로직은 없다.
API 전체: [docs/api.md](docs/api.md) (영어).

## 백엔드

셋 다 같은 상태 텐서와 같은 식을 쓰고, 호출 하나로 바꾼다.

- **CUDA 커널** -- `attach_stick(dyn)`.  처리량용.  스텝당 커널 한 번, substep 동안 상태를
  레지스터에 둔다.
- **torch, GPU** -- `F16Stick(..., device="cuda")`, CUDA 그래프로 감싸도 된다.  커널의 기준
  구현이고, 파이썬에서 모델을 읽거나 고칠 때 쓴다.
- **torch, CPU** -- `F16Stick(..., device="cpu")`.  GPU 가 없을 때.  스레드 하나로도 약 1,000 대
  부터 JSBSim 1 코어보다 빠르지만, JSBSim 을 모든 코어에 돌리면 여전히 JSBSim 이 빠르다.

PyTorch 는 텐서 입출력, CUDA 그래프, 기준 구현을 맡고, 커널을 컴파일하는 NVRTC 도 들고 온다
-- CUDA 툴킷도 C++ 컴파일러도 필요 없다.

## JSBSim 과의 정합

JSBSim 1.3.0 을 날리다가 한가운데 상태 전부(적분기 이력, FLCS 지연버퍼 포함)를 이 플랜트에
심고, 같은 조종 입력으로 **한 프레임만** 전진시켜 비교한다.  입력 13 가지 × 비행조건 4 가지
중 최대 오차, float64:

| 양 | torch 백엔드 | CUDA 커널 |
|---|---|---|
| 승강타·보조익·방향타 각 | 0 | 0 |
| u̇, v̇, ẇ (ft/s²) | 7.2e-6, 2.1e-5, 1.4e-4 | 같음 |
| ṗ, q̇, ṙ (rad/s²) | 1.2e-7, 1.4e-8, 4.4e-9 | 같음 |
| 추력 (lbf), N2 (%) | 3.4e-6, 1.6e-9 | 같음 |
| 질량·관성 (무작위 탱크 200 개) | 상대 3e-11 이하 | 같음 |
| 커널 대 torch, 모든 상태, 스텝마다 | -- | 상대 5.3e-14 |

진짜 버그 하나를 일부러 되살린 같은 대조(`--negative`)는 실패한다 -- 그래야 한다.
재현: `pip install jsbsim==1.3.0 && python -m jsbsim_f16_cuda.fdm_verify --check --fused`.
방법, float32·위도별 결과, 트림·질량 검증: [docs/verification.md](docs/verification.md).
긴 궤적은 정합의 척도가 아니다 -- 초기속도를 1e-6 kt 바꾸면 JSBSim 은 **자기 자신과도** 수백
초 안에 km 단위로 갈린다.

## 한계

- **기종 하나**: JSBSim 1.3.0 번들 F-16.  JSBSim 엔진이 아니다 -- 다른 기체 파일은 돌지 않는다.
- 착륙장치는 내린 채 고정(JSBSim 이 로드한 상태).  지면 접촉·바람·난류 없음, 표준대기뿐.
- 연료: 탱크 사이 이송·투하 없음.  연료가 다 떨어져도 엔진이 안 꺼진다 (JSBSim 은 끈다).
- `reset()` 은 수평 트림만.  JSBSim 트림 표(0 ~ 45,000 ft, 150 ~ 800 kt)를 보간한다.  스케줄이
  꺾이는 칸 사이에서는 트림한 기체가 10 초에 수십 ft 움직일 수 있다.
- 기준 위도 둘레의 평평한 지구 + 자전 항.  원점에서 수십 km 넘게 멀어지면 JSBSim(구면)과 벌어진다.
- 커널은 NVIDIA GPU 와 `integrator="jsbsim"` 이 필요하다.  float32 는 float64 와 반올림 수준으로 다르다.

## 문서 (영어)

- [docs/api.md](docs/api.md) -- `F16Stick`, `reset`, `step`, `state`, 조종간 규약, CUDA 그래프
- [docs/benchmarks.md](docs/benchmarks.md) -- 배치별 전체 측정과 재측정 방법
- [docs/verification.md](docs/verification.md) -- 한 프레임 대조 방법과 전체 결과
- [docs/porting_notes.md](docs/porting_notes.md) -- 옮기며 찾은 조용한 함정들

## 라이선스와 출처

GPL-3.0-or-later ([LICENSE](LICENSE)).  [JSBSim](https://github.com/JSBSim-Team/jsbsim)
(LGPL-2.1) 의 운동방정식·적분기·질량·엔진·보조량 계산과, 그 F-16 모델 `aircraft/f16/f16.xml`
(Erik Hofman, GPL) · `engine/F100-PW-229.xml` 의 비행제어 논리와 계수표를 옮겼다
(`f16_tables.npz` 는 이 파일들에서 뽑은 데이터다).  f16.xml 머리말대로 이 모델은 공개 자료로
만든 것이며 실제 제조사와 무관하다.
