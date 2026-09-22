# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written CUDA kernel implementation of JSBSim's F-16 flight model.

    from jsbsim_f16_cuda import F16Stick, attach_stick
    dyn = F16Stick(65536, device="cuda")      # state lives in torch tensors
    attach_stick(dyn)                         # dyn.step(stick, substeps) now runs the CUDA kernel

One CUDA thread flies one aircraft; one kernel launch advances every aircraft by
`substeps` physics frames (1/120 s each).  The kernel is CUDA C++ generated from the
model tables on first `attach_stick` and compiled at runtime with NVRTC (shipped with PyTorch),
so no CUDA toolkit or C++ compiler is needed.  PyTorch also provides a reference
implementation of the same equations (`F16Stick.step` without `attach_stick`), which
runs on CPU or GPU.

This is the F-16 model bundled with JSBSim 1.3.0 (f16.xml + F100-PW-229 + FLCS), not the
JSBSim engine: other aircraft files will not run.
"""
from .f16_core import F16Stick, TankMass, TrimGrid, gravity_j2_ms2  # noqa: F401

__version__ = "0.2.0"


def attach_stick(dyn):
    """Route `dyn.step` through the CUDA kernel (`fused_core.attach_stick`).  CUDA only."""
    from .fused_core import attach_stick as _attach
    return _attach(dyn)
