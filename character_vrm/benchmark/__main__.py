"""``python3 -m character_vrm.benchmark model.vrm [...] [--json out.json] [--pass named,blind,strip]``"""

from __future__ import annotations

import argparse
import json
import sys

from .runner import run_model


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _summary(res: dict) -> list[str]:
    lines = [f"== {res['model']}  (IOD {res['truth']['iod']}, truth: {', '.join(res['truth']['landmarks'])})"]
    for note in res["truth"]["notes"]:
        lines.append(f"   truth note: {note}")
    for name in ("named", "blind", "strip"):
        r = res.get(name)
        if not r:
            continue
        lm = r["landmarks"]
        h = r["humanoid"]
        ex = r["expressions"]
        ok = sum(1 for v in ex.values() if v == "ok")
        lines.append(f"-- {name}: landmark error / IOD " + ", ".join(f"{k}={v}" for k, v in lm.items()))
        lines.append(f"   humanoid {h['correct']}/{h['total']}  wrong={len(h['wrong'])} missing={len(h['missing'])}"
                     f"   expressions {ok}/{len(ex)}  hair={r['hair']}  face={r['face']}")
        bad = {k: v for k, v in ex.items() if v != "ok"}
        if bad and name != "strip":
            lines.append(f"   expression misses: {bad}")
        if h["wrong"]:
            lines.append(f"   humanoid wrong: {dict(list(h['wrong'].items())[:8])}")
        if name == "strip":
            for e, g in r.get("generation", {}).items():
                lines.append(f"   gen {e}: {g}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="character_vrm.benchmark")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--pass", dest="passes", default="named,blind,strip")
    args = ap.parse_args(argv)
    results = [run_model(m, tuple(args.passes.split(","))) for m in args.models]
    for r in results:
        print("\n".join(_summary(r)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(_clean(results), f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
