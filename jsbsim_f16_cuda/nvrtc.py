# SPDX-License-Identifier: GPL-3.0-or-later
"""CUDA C++ 소스를 런타임에 CUBIN 으로 -- NVRTC (torch 가 들고 오는 DLL) + 디스크 캐시.

CUDA 툴킷(nvcc)도 MSVC 도 필요 없다.  `torch.cuda._compile_kernel` 과 같은 일인데
그쪽은 `CUDA_HOME` 의 include 경로를 **무조건** 찾다가 툴킷이 없으면 죽는다.  우리
커널은 헤더를 안 쓰므로 그 줄만 뺐다.

    k = nvrtc.load(src, "fdm_step", ["-std=c++17"])
    k(grid=(g, 1, 1), block=(128, 1, 1), args=[n, tensor, ...])

실행은 torch 의 `_CudaKernel` 이다 -- **지금 스트림**에 `cuLaunchKernel` 하므로
`torch.cuda.graph` 캡처 안에서 그대로 잡힌다.  정수 인자는 C `int`, 텐서는
`data_ptr()` 이다 (빈 텐서 = 널 포인터).  파이썬 실수는 C `double` 로 넘어가니
커널 인자에 `float` 를 두지 마라.
"""
from __future__ import annotations

import ctypes
import hashlib
import os

import torch

_CACHE = os.path.join(os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__))),
                      "jsbsim_f16_cuda_nvrtc")


def _compile(src: str, name: str, opts: list[str]) -> bytes:
    from torch.cuda import _utils as U
    lib = U._get_gpu_rtc_library()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = f"--gpu-architecture=sm_{props.major}{props.minor}"
    options = [arch.encode()] + [o.encode() for o in opts]

    def check(res):
        if res != 0:
            s = ctypes.c_char_p()
            lib.nvrtcGetErrorString(res, ctypes.byref(s))
            raise RuntimeError(f"NVRTC: {s.value.decode() if s.value else res}")

    prog = ctypes.c_void_p()
    check(lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), f"{name}.cu".encode(),
                                 0, None, None))
    arr = (ctypes.c_char_p * len(options))(*options)
    res = lib.nvrtcCompileProgram(prog, len(options), arr)
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    if res != 0:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"커널 컴파일 실패 ({name}):\n{log.value.decode(errors='replace')}")
    size = ctypes.c_size_t()
    check(lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
    buf = ctypes.create_string_buffer(size.value)
    check(lib.nvrtcGetCUBIN(prog, buf))
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


def load(src: str, name: str, opts: list[str] | None = None):
    """`extern "C" __global__ void <name>(...)` 하나를 컴파일해 돌려준다 (캐시)."""
    from torch.cuda import _utils as U
    opts = list(opts or [])
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    key = hashlib.sha1("\0".join([src, name, *opts, str(torch.version.cuda),
                                  f"{props.major}{props.minor}"]).encode()).hexdigest()[:20]
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"{name}_{key}.cubin")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cubin = f.read()
    else:
        cubin = _compile(src, name, opts)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(cubin)
        os.replace(tmp, path)
    mod = U._cuda_load_module(cubin)
    return getattr(mod, name)
