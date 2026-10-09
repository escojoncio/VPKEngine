#!/usr/bin/env python3
"""The first instruction of a failing generated case whose result differs (native vs translated).

  bisect.py CASE...   (names as in tests/aot/cases, e.g. jaguar_base_00)
Prints, for each case, that instruction and the state lines that differ."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "tests" / "aot" / "cases"

def run(lines, head):
    tmp = CASES / "zz_bisect.s"
    tmp.write_text("\n".join(head + lines + ["    ret"]) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "tests/aot/run.py"), "zz_bisect"], capture_output=True, text=True)
    tmp.unlink()
    return r.returncode == 0, r.stdout + r.stderr

def diff(out):
    native, translated, cur = [], [], None
    for l in out.splitlines():
        if l.startswith("--- native"): cur = native; continue
        if l.startswith("--- translated"): cur = translated; continue
        if cur is not None and "=" in l: cur.append(l)
    return [f"{a}  !=  {b}" for a, b in zip(native, translated) if a != b] or [out[-600:]]

for case in sys.argv[1:]:
    text = (CASES / f"{case}.s").read_text().splitlines()
    start = text.index("_start:") + 1
    head = [l for l in text[:start]]
    body = [l for l in text[start:] if l.strip() and l.strip() != "ret"]
    # Units: an instruction with the flag clearing that follows it.
    units = []
    for l in body:
        if l.strip().startswith("pushfq") and units: units[-1].append(l)
        else: units.append([l])
    lo, hi = 0, len(units)  # the first n units fail for n = hi; pass for n = lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _ = run([x for u in units[:mid] for x in u], head)
        if ok: lo = mid
        else: hi = mid
    ok, out = run([x for u in units[:hi] for x in u], head)
    print(f"== {case}: {units[hi - 1][0].split('#', 1)[-1].strip()}")
    for d in diff(out)[:8]: print("   ", d)
