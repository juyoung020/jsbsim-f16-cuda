# Porting notes

Places where a port of JSBSim's F-16 runs without errors but gives different numbers.
Every item below was found by the frame-by-frame comparison
([verification.md](verification.md)); most of them move nothing you would notice in a
plot of a trajectory.

## Equations of motion

1. **Coriolis appears twice.**  JSBSim's `FGAccelerations::CalculateUVWdot` uses
   `-(pqr + 2 Ω) × uvw`.  With Ω only once, v̇ at 450 kt is off by 0.034 ft/s² — 26 times
   the whole v̇ in that condition.
2. **Centrifugal acceleration of the Earth's rotation: vertical part only.**
   `Ω² r cos²(lat)` (about 0.2 % of g) must be added along the local vertical.  The
   horizontal part is already absorbed in the geodetic "down" direction; adding it makes
   u̇ 36 times worse.
3. **Gravity magnitude is JSBSim's J2 model**, not a constant, evaluated at the
   reference latitude (`gravity_j2_ms2`, agrees with `accelerations/gravity-ft_sec2` to 1e-8).
4. **Product of inertia sign.**  Use `inertia/ixz-slugs_ft2` as is for J[0,2].  Following
   f16.xml's `negated_crossproduct_inertia="true"` literally makes angular accelerations
   1,700 times worse.
5. **Adams-Bashforth history slot `[0]` is the previous frame.**  JSBSim integrates
   velocity with AB2 (inertial frame, item 6) and position with AB3.  When seeding a mid-flight state, filling slots
   `[1]`, `[2]` only is 90 times worse than not seeding at all.

6. **Translational velocity is integrated in the inertial (local NED) frame, not in body
   axes.**  JSBSim's `FGPropagate` advances the inertial velocity with AB2 and rebuilds u, v, w
   from the *new* attitude.  Body-axis AB2 is the same equation in continuous time but not in
   discrete time: the stored past derivatives are in the body axes of their own frame, so while
   the aircraft rotates they are extrapolated along stale axes.  On JSBSim's own records in
   rapidly rolling flight, body-axis AB2 misses the next frame's velocity by up to 0.10 ft/s;
   NED AB2 matches to 2.4e-4 ft/s (the rest is the flat-Earth transport rate).  Stick inputs
   held for several frames hide this almost completely.

## Timing within a frame

7. **Controls act one frame late.**  A JSBSim frame starts with the integrator
   (Propagate), so the forces made by `set_controls(u)` enter the *next* integration.
   `F16Stick.step(u)` equals `set_controls(u); run()` exactly.
8. **The FLCS delays itself by one frame.**  Several of its inputs (alpha, rates, pilot
   load factor, Mach, calibrated airspeed) are last frame's values; the attitude inputs
   are this frame's.  Delaying again outside the FLCS only shows up at high load.
9. **Pilot load factor mixes ages**: last frame's specific force and angular
   acceleration, this frame's angular rate.  The other combinations agree at the median
   and disagree by up to 4 g at high load.
10. **Mass is evaluated with the fuel burned up to the previous frame**
   (`FGMassBalance` runs before `FGPropulsion`), and **the tank parallel-axis terms use
   the centre of gravity from one frame earlier still** (`FGPropulsion` computes them
   with the CG it had).  Without the second rule inertia is off by 1e-6 (relative);
   with it, 3e-11.
11. **The first frame after a trim snaps the engine** (N2 and fuel flow jump to the
    commanded value — JSBSim's trim step), then normal spooling resumes.

## Engine and fuel

12. **Fuel flow is rate-limited** (`FGTurbine::Seek`: +5,000 / −10,000 pph/s).  Thrust
    follows the throttle within a frame; fuel flow crawls.  While the engine spools the
    two differ by up to 40 times, while at steady state the plain `thrust × sfc` formula
    is exact — so steady-state sweeps never show the problem.
13. **Throttle is applied once.**  The FLCS output `fcs/throttle-pos-norm` is already
    0 – 2; the turbine takes `fcs/throttle-cmd-norm` (0 – 1) and doubles it itself.
    Feeding it the FLCS output doubles the throttle only in the middle of the range.

## Atmosphere and flaps

14. **Calibrated airspeed**: impact pressure uses the local static pressure, the inverse
    uses sea-level static pressure.  Mixing them is ~250 kt wrong at 20,000 ft and moves
    every FLCS gain schedule and the flap switch.
15. **The trailing-edge flap initial value is a switch, not a table.**  Interpolating it
    by airspeed gives a value that no branch produces (0.4966 instead of 1.0 at
    325 kt / 20,000 ft), and the flaps then sit half-deployed for 1.5 s.

## JSBSim itself (for the reference runs)

These are in [`jsbsim_f16_cuda/f16_reference.py`](../jsbsim_f16_cuda/f16_reference.py):
`run_ic()` / `do_trim()` do not start the engine; `run_ic()` does not clear the FCS
integrators (and `reset_to_initial_conditions` resets the tanks); attitude and rate
initial conditions must all be stated; read geodetic latitude, not geocentric (0.19° =
20 km); pass the JSBSim root directory explicitly.
