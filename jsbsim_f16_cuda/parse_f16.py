# SPDX-License-Identifier: GPL-3.0-or-later
"""f16.xml / F100-PW-229.xml -> f16_tables.npz  (한 번만 돌린다).

런타임은 XML 을 절대 파싱하지 않는다.  이 스크립트가 굳혀 놓은 npz 만 읽는다.

    python parse_f16.py            # jsbsim_f16_cuda/f16_tables.npz 를 다시 만든다
    python parse_f16.py --dump     # 만들고 구조를 사람이 읽게 찍는다

굳히는 것
---------
metrics       Sw, bw, cbar (ft2, ft, ft)  + AERORP 위치
aero          축(DRAG/SIDE/LIFT/ROLL/PITCH/YAW)별 function 목록.
              JSBSim 의 <function> 은 전부 <product> 안에 property / value /
              table 이 늘어선 꼴이라, 그 목록을 그대로 굳힌다.
engine        milthrust/maxthrust/bypassratio/idlen2/maxn2 + Idle/Mil/Aug 2D 표

주의: JSBSim 의 aero/coefficient/* 는 이름과 달리 **무차원 계수가 아니다.**
product 안에 aero/qbar-psf * metrics/Sw-sqft 가 이미 들어 있어 값은 lbf
(모멘트축은 lbf*ft) 다.  그대로 두는 편이 대조가 쉬워 그대로 굳힌다.
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _jsbsim_root() -> str:
    import jsbsim
    return os.path.dirname(jsbsim.__file__)


# ---------------------------------------------------------------- table parse

def _numbers(text: str) -> list[list[float]]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(t) for t in line.split()])
    return rows


def parse_table(el: ET.Element) -> dict:
    """<table> -> {'ndim', 'row', 'col', 'data', 'row_var', 'col_var'}."""
    ivars = el.findall("independentVar")
    row_var = col_var = None
    for iv in ivars:
        lookup = (iv.get("lookup") or "row").strip()
        if lookup == "row":
            row_var = iv.text.strip()
        elif lookup == "column":
            col_var = iv.text.strip()
        else:
            raise ValueError(f"table lookup {lookup!r} 는 아직 안 다룬다")
    rows = _numbers(el.find("tableData").text)

    if col_var is None:                                   # 1D
        arr = np.asarray(rows, dtype=np.float64)
        assert arr.shape[1] == 2, arr.shape
        return dict(ndim=1, row=arr[:, 0].copy(), col=np.zeros(0),
                    data=arr[:, 1].copy(), row_var=row_var, col_var="")

    # 2D: 첫 줄이 column breakpoints, 이후 각 줄이 [row_bp, v0, v1, ...]
    col = np.asarray(rows[0], dtype=np.float64)
    body = np.asarray(rows[1:], dtype=np.float64)
    assert body.shape[1] == col.size + 1, (body.shape, col.size)
    return dict(ndim=2, row=body[:, 0].copy(), col=col,
                data=body[:, 1:].copy(), row_var=row_var, col_var=col_var)


# ------------------------------------------------------------ function parse

def parse_function(el: ET.Element) -> dict:
    """<function> -> {'name', 'factors': [...]}  (product 또는 단독 table)."""
    name = el.get("name")
    prod = el.find("product")
    kids = list(prod) if prod is not None else [
        k for k in el if k.tag in ("table", "property", "value")]

    factors = []
    for k in kids:
        if k.tag == "property":
            factors.append(dict(kind="property", prop=k.text.strip()))
        elif k.tag == "value":
            factors.append(dict(kind="value", value=float(k.text)))
        elif k.tag == "table":
            factors.append(dict(kind="table", table=parse_table(k)))
        elif k.tag == "description":
            pass
        else:
            raise ValueError(f"{name}: <{k.tag}> 는 아직 안 다룬다")
    desc = el.findtext("description", "").strip()
    return dict(name=name, factors=factors, description=desc)


# ------------------------------------------------------------------- flatten
#
# npz 는 중첩 dict 를 못 담으므로 평평한 키로 편다.
#   aero.<axis>.<i>.name            function 이름
#   aero.<axis>.<i>.factor.<j>.kind 'property' | 'value' | 'table'
#   ... .prop / .value / .table.{row,col,data,row_var,col_var,ndim}

def flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}", v, out)
    elif isinstance(obj, (list, tuple)):
        out[f"{prefix}.__len__"] = np.int64(len(obj))
        for i, v in enumerate(obj):
            flatten(f"{prefix}.{i}", v, out)
    elif isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif isinstance(obj, str):
        out[prefix] = np.str_(obj)
    elif isinstance(obj, (int, np.integer)):
        out[prefix] = np.int64(obj)
    else:
        out[prefix] = np.float64(obj)


def unflatten(npz) -> dict:
    """flatten 의 역."""
    root: dict = {}
    for key in npz.files:
        parts = key.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = npz[key]
    return _listify(root)


def _listify(node):
    if not isinstance(node, dict):
        return node
    if "__len__" in node:
        n = int(node["__len__"])
        return [_listify(node[str(i)]) for i in range(n)]
    return {k: _listify(v) for k, v in node.items()}


# ---------------------------------------------------------------------- main

def build(aircraft_xml: str, engine_xml: str) -> dict:
    root = ET.parse(aircraft_xml).getroot()

    m = root.find("metrics")
    aerorp = m.find("location[@name='AERORP']")
    metrics = dict(
        Sw_sqft=float(m.findtext("wingarea")),
        bw_ft=float(m.findtext("wingspan")),
        cbar_ft=float(m.findtext("chord")),
        aerorp_in=np.array([float(aerorp.findtext("x")),
                            float(aerorp.findtext("y")),
                            float(aerorp.findtext("z"))]),
    )

    aero_el = root.find("aerodynamics")
    helpers = [parse_function(f) for f in aero_el.findall("function")]
    axes = {}
    for ax in aero_el.findall("axis"):
        axes[ax.get("name")] = [parse_function(f) for f in ax.findall("function")]

    e = ET.parse(engine_xml).getroot()
    def ef(tag, default=None):
        t = e.findtext(tag)
        return default if t is None else float(t)
    engine = dict(
        milthrust=ef("milthrust"), maxthrust=ef("maxthrust"),
        bypassratio=ef("bypassratio"),
        idlen1=ef("idlen1", 30.0), idlen2=ef("idlen2", 60.0),
        maxn1=ef("maxn1", 100.0), maxn2=ef("maxn2", 100.0),
        augmented=ef("augmented", 0.0), augmethod=ef("augmethod", 0.0),
        tables={f.get("name"): parse_table(f.find("table"))
                for f in e.findall("function")},
    )

    return dict(metrics=metrics, helpers=helpers, axes=axes, engine=engine)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "f16_tables.npz"))
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    root = _jsbsim_root()
    model = build(os.path.join(root, "aircraft", "f16", "f16.xml"),
                  os.path.join(root, "engine", "F100-PW-229.xml"))

    flat: dict = {}
    flatten("metrics", model["metrics"], flat)
    flatten("helpers", model["helpers"], flat)
    flatten("axes", model["axes"], flat)
    flatten("engine", model["engine"], flat)
    np.savez_compressed(args.out, **flat)
    print(f"{args.out}  ({os.path.getsize(args.out)/1024:.1f} KB, {len(flat)} keys)")

    if args.dump:
        props, tvars = set(), set()
        for axis, fns in model["axes"].items():
            print(f"\n=== {axis}  ({len(fns)} functions)")
            for fn in fns:
                bits = []
                for f in fn["factors"]:
                    if f["kind"] == "property":
                        bits.append(f["prop"]); props.add(f["prop"])
                    elif f["kind"] == "value":
                        bits.append(f"{f['value']:g}")
                    else:
                        t = f["table"]
                        if t["ndim"] == 1:
                            bits.append(f"T1[{t['row_var']}]({t['row'].size})")
                            tvars.add(t["row_var"])
                        else:
                            bits.append(f"T2[{t['row_var']} x {t['col_var']}]"
                                        f"({t['row'].size}x{t['col'].size})")
                            tvars.add(t["row_var"]); tvars.add(t["col_var"])
                print(f"  {fn['name']:28s} = " + " * ".join(bits))
        print("\n--- properties used:"); print("   " + "\n   ".join(sorted(props)))
        print("--- table independent vars:"); print("   " + "\n   ".join(sorted(tvars)))
        print("--- engine:", {k: v for k, v in model["engine"].items() if k != "tables"})
        for n, t in model["engine"]["tables"].items():
            print(f"    {n}: {t['row_var']} x {t['col_var']} "
                  f"({t['row'].size}x{t['col'].size})")


if __name__ == "__main__":
    main()
