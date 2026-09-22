# API

```python
from jsbsim_f16_cuda import F16Stick, attach_stick
```

## `F16Stick(n, device="cuda", dtype=torch.float32, *, lat0_deg=0.0, refuel=False, fuel_lbs=None, gravity_ms2=None, integrator="jsbsim")`

`n` F-16s.  Every state is an `(n, ...)` tensor on `device`, updated in place.

| argument | meaning |
|---|---|
| `lat0_deg` | reference latitude of the local tangent plane; sets the Earth-rotation terms and the J2 gravity magnitude |
| `refuel` | JSBSim `propulsion/refuel` (100 lb/s shared equally among tanks that are not full).  Default off: fuel is burned normally |
| `fuel_lbs` | initial fuel of the 4 tanks [lb].  Default f16.xml: (1,500, 1,500, 0, 0).  Capacities (3,486, 3,486, 2,991, 2,991) |
| `gravity_ms2` | override the gravity magnitude at 20,000 ft (default: JSBSim J2 value at `lat0_deg`) |
| `integrator` | `"jsbsim"` (JSBSim's default integrator set; required by the CUDA kernel), `"euler"`, `"rk4"` |

## `attach_stick(dyn)`

Routes `dyn.step` through the CUDA kernel.  The kernel reads and writes the same state
tensors in place, so `reset`, `state()` and everything else keep working.  CUDA only,
`integrator="jsbsim"` only.  The first call generates the CUDA C++ source from the model
tables and compiles it with NVRTC (a few seconds); the binary is cached under
`%TEMP%/jsbsim_f16_cuda_nvrtc` (`$TEMP` or next to the package on other systems).
float32 and float64 compile separately.  `jsbsim_f16_cuda.fused_core.detach(dyn)` undoes it.

## `reset(pos_ned, psi, vt_ms, fuel_lbs=None, mask=None) -> ok`

Start in **level trimmed flight** (the state JSBSim's `do_trim(1)` produces).

| argument | shape | unit |
|---|---|---|
| `pos_ned` | `(n, 3)` | m: north, east, **down** (down = −altitude above sea level) |
| `psi` | `(n,)` | rad: true heading (north 0, east +π/2) |
| `vt_ms` | `(n,)` | m/s: true airspeed |
| `fuel_lbs` | `(n, 4)` | lb per tank |
| `mask` | `(n,)` bool | reset only these aircraft (no host sync — works inside a CUDA graph) |

Trim attitude, pitch trim, trim throttle, N2 and the FLCS delay buffers come from a table
measured with JSBSim (`f16_trim.npz`: 0 – 45,000 ft × 150 – 800 kt × 7 fuel loads),
trilinearly interpolated.  `ok` is False where the point is outside the table or any
surrounding grid point failed to trim (e.g. slow and high: level flight impossible).

## `step(stick, substeps=1)`

`stick` is `(n, 4)` = (aileron, elevator, rudder, throttle).

| channel | range | positive means | JSBSim property |
|---|---|---|---|
| aileron | [−1, 1] | roll right | `fcs/aileron-cmd-norm` = aileron |
| elevator | [−1, 1] | pull (nose up) | `fcs/elevator-cmd-norm` = −elevator |
| rudder | [−1, 1] | yaw right | `fcs/rudder-cmd-norm` = −rudder |
| throttle | [0, 1] | 1 = full afterburner | `fcs/throttle-cmd-norm` = throttle |

One `step(u)` equals JSBSim's `set_controls(u); run()`.  JSBSim starts a frame with the
integrator, so the controls you set become forces in the *next* integration; this plant
holds the previous controls and applies them one frame late in the same way.  After
`step`, position, attitude, velocity and rates are at the same instant as JSBSim right
after `run()`.  `substeps=k` repeats the same controls for k frames (one kernel launch).

## `state() -> dict`

| key | shape | unit |
|---|---|---|
| `pos_ned` | `(n, 3)` | m |
| `quat` | `(n, 4)` | local NED → body quaternion (w, x, y, z), same as JSBSim |
| `euler` | `(n, 3)` | rad: (phi bank, theta pitch, psi heading), 3-2-1 |
| `uvw` | `(n, 3)` | m/s, body axes |
| `vel_ned` | `(n, 3)` | m/s |
| `pqr` | `(n, 3)` | rad/s, body axes |
| `alpha`, `beta` | `(n,)` | rad |
| `n2` | `(n,)` | % |
| `fuel_lbs` | `(n, 4)` | lb |
| `n_pilot` | `(n, 3)` | g, pilot load factor with JSBSim's `accelerations/n-pilot-*-norm` sign (z = −1 in level flight) |

`state()` returns copies.  The live tensors are `dyn.rb.pos_ned`, `dyn.rb.uvw`,
`dyn.rb.quat`, `dyn.rb.pqr`, `dyn.n2`, `dyn.fuel`, `dyn.flcs.*`.

## CUDA graphs

State is only ever updated in place (addresses never change), so the torch backend can
be captured in a CUDA graph; the kernel backend is a single launch and gains little.

```python
stick_buf = torch.zeros(N, 4, device="cuda")
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        dyn.step(stick_buf, 6)                 # warm-up before capture
torch.cuda.current_stream().wait_stream(side)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    dyn.step(stick_buf, 6)
for _ in range(1000):
    stick_buf.copy_(policy(...))               # write inputs into the captured buffer
    g.replay()
```

`reset(..., mask=)` has no host synchronisation and can be captured too.
