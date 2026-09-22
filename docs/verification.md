# Verification against JSBSim

Everything here is reproducible with the commands at the end (`pip install jsbsim==1.3.0`).
The reference is JSBSim 1.3.0 itself, driven through `jsbsim_f16_cuda.f16_reference`
with JSBSim's standard behaviour (trim, then hold the trim controls; fuel burns normally;
no refuelling).

## 1. Why frame by frame, not trajectories

Comparing positions after a minute of flight cannot tell you *which* term is wrong.
Flight is chaotic: JSBSim compared against itself, with the initial airspeed changed by
1e-6 kt, drifts apart by kilometres within a few hundred seconds under random stick input.
And a per-layer check ("feed JSBSim's inputs, compare JSBSim's outputs") can pass while the
equation that turns forces into motion is wrong, because reference and subject share it.

During this port, fixing the bugs listed in [porting_notes.md](porting_notes.md) cut the
one-frame error by 300 – 2,000 times but the 60-second random-input trajectory error by only
about 20 %.  Trajectory metrics are not a fidelity test.

## 2. Method

> Take JSBSim in mid-flight, copy its **entire** state into the batched plant, advance
> **one** frame (1/120 s), and compare every intermediate quantity.  Nothing accumulates,
> so whatever differs is a modelling difference.

Both engines get the same raw stick inputs every frame (13 input programs — steps,
doublets, ramps, full deflection, random — × 4 flight conditions from 320 kt / 5,000 ft to
600 kt / 30,000 ft).  Compared per frame: the 6 control surfaces, thrust, N2, fuel, mass,
inertia and CG, alpha / beta / Mach / dynamic pressure / calibrated airspeed, pilot load
factor, aerodynamic and total forces and moments, the 6 state derivatives, and the
integrated state.

**Frame alignment.**  After one JSBSim `run()`, row k holds (state S_k, forces computed at
S_k); those forces take S_k to S_{k+1}.  Seeding row W therefore pairs GPU frame j's forces
with JSBSim row W+j and GPU frame j's integrated state with row W+j+1.  Pairing forces one
row off makes an exact FCS look 1e-3 wrong.

**State is more than position and velocity.**  Seeded from the rows around W:

| what | row |
|---|---|
| position, velocity, attitude, rates | W |
| Adams-Bashforth history (inertial-frame velocity and position derivatives) | W−1, W−2, W−3 |
| FLCS delay buffers (9) and surface positions (3) | W−1 |
| N2, fuel flow | W−1 |
| previous specific force and angular acceleration (pilot load factor) | W−1 |
| tank contents | W−1 |
| CG used for the tank parallel-axis terms | from tanks at W−2 |
| pitch trim | W |

## 3. One-frame results

Maximum |error| over the 52 combinations.  The CUDA kernel column is the same
comparison run through the kernel (`--fused`); it matches the torch backend to the digits
shown.

| quantity | torch, float64, lat 0 | CUDA kernel, float64, lat 0 | float64, lat 60 | float32, lat 0 | unit |
|---|---|---|---|---|---|
| u̇ | 7.2e-6 | 7.2e-6 | 4.0e-4 | 1.1e-5 | ft/s² |
| v̇ | 2.1e-5 | 2.1e-5 | 1.8e-5 | 2.3e-5 | ft/s² |
| ẇ | 1.4e-4 | 1.4e-4 | 1.8e-4 | 1.5e-4 | ft/s² |
| ṗ | 1.2e-7 | 1.2e-7 | 1.2e-7 | 2.3e-6 | rad/s² |
| q̇ | 1.4e-8 | 1.4e-8 | 1.4e-8 | 1.2e-6 | rad/s² |
| ṙ | 4.4e-9 | 4.4e-9 | 4.5e-9 | 2.6e-7 | rad/s² |
| elevator, aileron, rudder | **0** | **0** | 1e-17 | 4e-8 | rad |
| thrust | 3.4e-6 | 3.4e-6 | 3.4e-6 | 2.9e-3 | lbf (of 10³ – 2·10⁴) |
| N2 | 1.6e-9 | 1.6e-9 | 1.6e-9 | 6.7e-6 | % |

The largest remaining term, ẇ, is the 1e-8 (relative) difference in the standard
atmosphere plus the flat-Earth approximation (constant Earth radius in the inverse-square
gravity).  At latitude 60 the u̇ residual grows to 4e-4 ft/s² (the horizontal centrifugal
component is absorbed in the geodetic vertical, which is exact only at the reference
point).  float32 differences are rounding (eps 1.2e-7); `--check` thresholds are set for
float64, so a float32 run reports four items above threshold by design.

**What these input programs do not exercise.**  They hold each stick input for at least 8
frames, so the integrator history is smooth.  One discrepancy was found only on rapidly
rolling flight: integrating velocity in body axes instead of the inertial frame
([porting_notes.md](porting_notes.md), item 6) — up to 0.10 ft/s per frame there, yet around
1e-6 in this table.  It is fixed; the numbers above did not change.

**Negative control.**  `fdm_verify --negative` deliberately removes one real bug fix (the
vertical centrifugal term, 0.2 % of g).  The check then fails on u̇, v̇ and ẇ (ẇ off by
0.11 ft/s², 100 times its threshold), for both the torch backend and the kernel.  A
comparison that could not catch that would be useless.

## 4. CUDA kernel vs torch backend

`f16_check fused`: 512 aircraft, random stick, 1 – 6 frames per call, two configurations
(tanks without refuelling at latitude 0; with refuelling at latitude 60).  Relative error =
|difference| / max |value| of that state, over all state tensors.

| | re-synchronised every step (max) | free flight, 60 steps |
|---|---|---|
| float64 | 5.3e-14 | 4.5e-13 |
| float32 | 8.5e-5 | 3.6e-4 |

The kernel is not bit-identical to torch: torch rounds after every operation and sums the
aerodynamic axes with a matrix product.  `--fmad=false` keeps the kernel from fusing
multiply-adds so it rounds like torch as far as possible.

## 5. Mass, trim table, coefficient tables

**Mass** (`f16_check mass`): the four tanks are point masses computed with JSBSim's own
rules and order.  200 random tank loads: weight 4e-12 lb, CG 3e-14 in, inertia ≤ 3e-11
slug·ft².  20 s of random throttle with refuelling off / on, full tanks, and tanks running
dry: per-tank fuel error **0**.

**Trim table** (`f16_trim_build --check`): `reset()` interpolates a table of JSBSim
`do_trim(1)` results over 0 – 45,000 ft (2,500 ft) × 150 – 800 kt (10 kt) × 7 fuel loads
(500 – 12,954 lb): 8,778 trims, 80.7 % converged.  At 200 random points *between* grid
points:

| | median | max |
|---|---|---|
| trim pitch | 5.7e-5 rad | 9.7e-4 rad |
| pitch trim command | 2.0e-4 | 1.2e-2 |
| trim throttle | 3.6e-4 | 7.2e-3 |
| N2 | 5.1e-3 % | 0.96 % |

Holding the trim controls for 10 s from those points changes altitude by a median 2.0 ft
(JSBSim itself: 1.8 ft — its trim is not perfectly still either).  The worst case sits
between grid cells where a schedule bends: 32,276 ft / 465 kt drifts +38 ft in 10 s
(JSBSim +1.4 ft).

**Coefficient tables** (`validate_aero`, 2,000 samples across the envelope with JSBSim's
own state as input): aerodynamic forces and moments median relative error 6e-8 – 1.2e-7
(max within 1e-4), standard atmosphere 5.7e-7, thrust table 5.7e-6, spool step response
N2 within 1e-4 %.

## 6. Reproduce

```bash
python -m jsbsim_f16_cuda.fdm_verify --check              # torch backend, float64, latitude 0
python -m jsbsim_f16_cuda.fdm_verify --check --fused      # CUDA kernel
python -m jsbsim_f16_cuda.fdm_verify --check --lat 60
python -m jsbsim_f16_cuda.fdm_verify --dtype float32      # table only (thresholds are float64)
python -m jsbsim_f16_cuda.fdm_verify --negative           # must report the injected bug
python -m jsbsim_f16_cuda.f16_check fused                 # kernel vs torch
python -m jsbsim_f16_cuda.f16_check mass
python -m jsbsim_f16_cuda.f16_trim_build --check
python -m jsbsim_f16_cuda.validate_aero
python -m jsbsim_f16_cuda.rbdyn_validate                  # integrator alone, JSBSim forces injected
```

`--prog` / `--cond` pick one input program / flight condition, `--trace wdot,cgz` prints a
quantity frame by frame.  The tool output is in Korean (the project's working language);
the numbers are the same.
