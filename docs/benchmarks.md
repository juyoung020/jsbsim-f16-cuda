# Benchmarks

All simulator numbers are in **aircraft-frames per second**: one frame is 1/120 s of flight
for one aircraft (JSBSim's default physics rate).  They measure **the physics only**; for
what that means for a whole training loop see
[Simulator vs. end-to-end training](#simulator-vs-end-to-end-training).  Raw data:
[bench.json](bench.json); figure: [images/throughput.png](images/throughput.png).

## Conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB), no other compute jobs during the GPU runs (utilisation 0 – 16 % before and after) |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 hardware threads; light background load (5 – 15 %) during the CPU runs |
| software | Windows 11, Python 3.11, torch 2.11.0+cu128, JSBSim 1.3.0 (pip) |
| this port | float32 unless stated; `step(stick, substeps=6)` per call (6 frames); each aircraft holds its own random stick and the fleet is re-trimmed every 10 s of flight (excluded from timing) so it keeps manoeuvring inside the envelope |
| JSBSim | one `FGFDMExec` per process, trim at 20,000 ft / 450 kt then hold the trim controls; re-trim every 100 s of flight (excluded); processes started together behind a barrier |
| timing | ≥ 3 s per measurement after warm-up; GPU numbers are the better of two runs (they differed by < 5 %) |

## Summary (physics only)

| backend | best configuration | aircraft-frames/s | cost per aircraft-frame | per 6-frame step | vs JSBSim 1 core, physics only |
|---|---|---|---|---|---|
| JSBSim 1.3.0, `run()` only | 1 process | 122 k | 8.2 µs | 49 µs | 1× |
| JSBSim 1.3.0, `run()` only | 32 processes | 2.38 M | 0.42 µs | 2.5 µs | 20× |
| torch CPU backend | 16,384 aircraft, 16 threads, float32 | 0.94 M | 1.1 µs | 6.4 µs | 7.7× |
| torch GPU backend, eager | 262,144 aircraft | 22.2 M | 45 ns | 270 ns | 182× |
| torch GPU backend, CUDA graph | 262,144 aircraft | 39.9 M | 25 ns | 150 ns | 327× |
| **CUDA kernel** | 262,144 aircraft | **3.09 G** | **0.32 ns** | **1.9 ns** | **25,300×** |

Cost per aircraft-frame = 1 / throughput, i.e. amortised over the whole batch (for JSBSim,
over the processes).  It is the number to plug into the formula in the next section.

Physics only, the kernel is **1,300×** JSBSim running on all 32 hardware threads of this
CPU.  3.09 G aircraft-frames/s is 25.8 million aircraft at the real-time rate of 120 frames
per second.

## CUDA kernel and torch GPU backend, by batch size

| aircraft | torch eager | torch + CUDA graph | CUDA kernel | kernel / graph |
|---|---|---|---|---|
| 1,024 | 0.10 M | 0.85 M | 169 M | 199× |
| 4,096 | 0.40 M | 3.23 M | 673 M | 208× |
| 16,384 | 1.65 M | 11.4 M | 2.34 G | 204× |
| 65,536 | 6.58 M | 29.3 M | 2.91 G | 99× |
| 131,072 | 12.6 M | 37.0 M | 3.04 G | 82× |
| 262,144 | 22.2 M | 39.9 M | 3.09 G | 77× |

The kernel saturates this GPU from about 65,536 aircraft.  The torch backend launches many
small kernels per frame; a CUDA graph removes the launch overhead but not the memory traffic
of writing every intermediate to global memory, which the kernel avoids by keeping each
aircraft's state in registers.

## JSBSim, by process count

| processes | `run()` only | per process | with I/O | per process |
|---|---|---|---|---|
| 1 | 122 k | 122 k | 102 k | 102 k |
| 8 | 832 k | 104 k | 575 k | 72 k |
| 16 | 1.40 M | 87 k | 1.01 M | 63 k |
| 32 | 2.38 M | 74 k | 1.58 M | 50 k |

"With I/O" writes the 4 controls and reads 12 state properties through the Python binding
every frame — what it costs to drive JSBSim from Python, for example in a reinforcement
learning loop.  Throughput per process falls as more processes share the chip's clocks,
caches and memory.

## torch CPU backend, by batch size

| aircraft | float32, 1 thread | float32, 16 threads | float64, 1 thread | float64, 16 threads |
|---|---|---|---|---|
| 1 | 349 | 413 | 459 | 455 |
| 64 | 24 k | 19 k | 27 k | 20 k |
| 1,024 | 262 k | 217 k | 245 k | 202 k |
| 16,384 | 645 k | 936 k | 478 k | 589 k |

Where it crosses JSBSim: per thread, the torch CPU backend is slower than one JSBSim core
below a few hundred aircraft (Python and operator overhead dominate — one aircraft is 350×
slower), passes it between 64 and 1,024 aircraft, and reaches 5.3× at 16,384 aircraft
(float32).  With all 16 threads it reaches 0.94 M, still **below JSBSim run on every core
(2.38 M)**.  On a CPU-only machine, running JSBSim itself in parallel processes is the
faster choice; the torch CPU backend is there as the reference implementation and for
machines without a GPU.

## Simulator vs. end-to-end training

The ratios above are for **the simulator alone**.  A training loop also runs the policy,
computes observations and rewards, and updates the network; this port does not make those
faster, and their cost depends entirely on your method.  So there is no single "training is
N× faster" number.  What carries over is Amdahl's law.  With `T_sim` the simulator time per
iteration (frames per iteration × cost per aircraft-frame from the table above) and `T_rest`
everything else:

    speedup = (T_sim_old + T_rest) / (T_sim_new + T_rest)

If the simulator is a fraction `p` of your current training time, the end-to-end speedup is
at most `1 / (1 − p)`, however fast the simulator becomes.

**One example, not a guarantee.**  A private PPO pipeline built on the same kernel: PPO,
1024×2 MLP policy, about 65,500 parallel two-aircraft episodes, 96 decision steps of 6
physics frames per iteration (≈ 6.3 M decision steps), 4 epochs, the same RTX 5070 Ti:

| physics backend in the same pipeline | one training iteration |
|---|---|
| torch backend | approx. 19.0 s |
| CUDA kernel | approx. 15.6 s (−18 %) |

The two numbers were measured at different times on the same machine, hence "approx.".
With the kernel, the physics inside one iteration takes about 26 ms (0.27 ms per decision
step × 96), about 0.2 % of the iteration; the rest is policy inference, observation and
reward computation, and gradient updates.  That pipeline adds a control layer on top of the
plant, compiled into the same kernel, so its physics cost is in the same range as the numbers
above.  Read through the formula: the two iteration times imply the torch physics took about
3.4 s of the 19.0 s (p ≈ 18 %; inferred, not measured separately), so the kernel can at best
give 1 / (1 − 0.18) ≈ 1.22 — which is the measured 19.0 / 15.6.  **When the simulator
becomes tens of thousands of times faster, end-to-end training only speeds up by the share the
simulator used to take — the bottleneck moves to the learning side, and the simulator cost all
but disappears.**

## Caveats

- The kernel and torch runs are float32; JSBSim computes in float64.  float64 changes the
  torch CPU numbers little (table above); for float32-vs-float64 accuracy see
  [verification.md](verification.md).
- Six frames per call is a typical control rate (20 Hz); other `substeps` values were not
  measured here.
- One GPU and one CPU model; other hardware will scale differently.

## Reproduce

```bash
pip install jsbsim==1.3.0 matplotlib
python examples/bench.py --all --json docs/bench.json     # about 10 minutes
python examples/plot_bench.py docs/bench.json docs/images/throughput.png
```

`--cpu`, `--jsbsim`, `--no-gpu`, `--sizes`, `--procs`, `--threads` select parts.
