# SPDX-License-Identifier: GPL-3.0-or-later
"""JSBSim `FGTable` 의 보간을 torch 배치로.

JSBSim 의 표는 **끝에서 외삽하지 않고 물린다(clamp)**.  1D `FGTable::GetValue`
는 키가 첫 브레이크포인트보다 작으면 첫 값을, 마지막보다 크면 마지막 값을
그대로 돌려준다.  2D 는 행·열 보간계수를 각각 [0,1] 로 자른다.  여기서도
`t.clamp(0,1)` 로 똑같이 한다 -- 외삽하면 alpha 45 도 위에서 조용히 갈라진다.

빠르게 만드는 방법은 **같은 독립변수를 쓰는 표를 하나로 쌓는 것**이다.
f16 의 1D 표 중 alpha 짜리 21 개는 브레이크포인트가 전부 같고, mach 짜리는
길이가 제각각이지만 **합집합 격자로 다시 샘플링하면 값이 정확히 보존된다**
(구간선형 함수를 자기 브레이크포인트를 포함하는 더 촘촘한 격자에서 다시
샘플링하는 것이므로 오차가 0 이다).  그래서 searchsorted 를 표마다가 아니라
독립변수마다 한 번만 부른다.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------- numpy (빌드)

def eval1_np(bp: np.ndarray, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """JSBSim 1D 조회 (clamp).  격자 합치기에만 쓴다."""
    i1 = np.clip(np.searchsorted(bp, x, side="left"), 1, bp.size - 1)
    i0 = i1 - 1
    span = bp[i1] - bp[i0]
    t = np.where(span != 0.0, (x - bp[i0]) / np.where(span != 0.0, span, 1.0), 1.0)
    t = np.clip(t, 0.0, 1.0)
    return data[i0] + t * (data[i1] - data[i0])


def eval2_np(rbp, cbp, data, r, c) -> np.ndarray:
    """JSBSim 2D 조회 (양축 clamp).  격자 합치기에만 쓴다."""
    out = np.empty((r.size, c.size))
    for a, rv in enumerate(r):
        i1 = int(min(max(np.searchsorted(rbp, rv, side="left"), 1), rbp.size - 1))
        i0 = i1 - 1
        span = rbp[i1] - rbp[i0]
        t = 1.0 if span == 0.0 else min(max((rv - rbp[i0]) / span, 0.0), 1.0)
        row = data[i0] + t * (data[i1] - data[i0])
        out[a] = eval1_np(cbp, row, c)
    return out


def regrid1(bp, data, new_bp) -> np.ndarray:
    return eval1_np(bp, data, new_bp)


def regrid2(rbp, cbp, data, new_r, new_c) -> np.ndarray:
    return eval2_np(rbp, cbp, data, new_r, new_c)


# --------------------------------------------------------------- torch (런타임)

def bracket(bp: torch.Tensor, x: torch.Tensor):
    """(i0, t) -- x 가 들어가는 구간의 아래 인덱스와 [0,1] 로 물린 보간계수."""
    i1 = torch.searchsorted(bp, x.contiguous().detach(), right=False)
    i1 = i1.clamp_(1, bp.numel() - 1)
    i0 = i1 - 1
    lo = bp[i0]
    t = ((x - lo) / (bp[i1] - lo)).clamp_(0.0, 1.0)
    return i0, t


def interp1_stack(bp: torch.Tensor, data: torch.Tensor, x: torch.Tensor):
    """표 K 개를 한 번에.  bp (N,), data (K, N), x (B,)  ->  (B, K)."""
    i0, t = bracket(bp, x)
    lo = data.index_select(1, i0)            # (K, B)
    hi = data.index_select(1, i0 + 1)
    return torch.addcmul(lo, hi - lo, t).transpose(0, 1)


def interp2_stack(rbp, cbp, data, r, c):
    """표 K 개를 한 번에.  data (K, R, C), r (B,), c (B,)  ->  (B, K)."""
    K, R, C = data.shape
    ri, rt = bracket(rbp, r)
    ci, ct = bracket(cbp, c)
    flat = data.reshape(K, R * C)
    base = ri * C + ci
    g00 = flat.index_select(1, base)
    g01 = flat.index_select(1, base + 1)
    g10 = flat.index_select(1, base + C)
    g11 = flat.index_select(1, base + C + 1)
    c0 = torch.addcmul(g00, g10 - g00, rt)
    c1 = torch.addcmul(g01, g11 - g01, rt)
    return torch.addcmul(c0, c1 - c0, ct).transpose(0, 1)
