#!/usr/bin/env python3
"""Where the host's x87 and runtime/vp_x87.c differ by 1 ulp, which one is the correctly rounded result?
Evidence that the difference is Intel's microcode, not an error of ours (it cannot be copied without the microcode).

  X87_DUMP=300 ./diff 20000 1 2> dump.txt >/dev/null ; python3 -I x87_trans_ref.py dump.txt

The reference is bc (scale 90) rounded once to 64 bits (nearest even). Prints, per instruction, how many of the
differing results had the host / ours / neither equal to the correctly rounded value.
"""
import subprocess
import sys
from fractions import Fraction

SC = 90

def val(m, se):
    e = (se & 0x7FFF) - 16383 - 63
    v = Fraction(m) * (Fraction(2) ** e)
    return -v if se & 0x8000 else v

def bcnum(fr):
    # exact decimal with SC digits is enough: arguments are dyadic; scale the numerator
    n, d = fr.numerator, fr.denominator
    return f"({n}/{d})"

def run_bc(expr):
    r = subprocess.run(["bc", "-l"], input=f"scale={SC}\n{expr}\n", capture_output=True, text=True, check=True)
    return Fraction(r.stdout.replace("\\\n", "").strip())

def round64(fr):
    if fr == 0:
        return (0, 0)
    sg = fr < 0
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    # a in [2^e, 2^(e+1)): mantissa = a * 2^(63-e)
    q = a * Fraction(2) ** (63 - e)
    m = q.numerator // q.denominator
    rem = q - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m & 1):
        m += 1
    if m == 1 << 64:
        m >>= 1
        e += 1
    return (m, (e + 16383) | (0x8000 if sg else 0))

def ref(op, a, b):
    A, B = bcnum(val(*a)), bcnum(val(*b))
    if op == "fsin": return run_bc(f"s({A})")
    if op == "fcos": return run_bc(f"c({A})")
    if op == "fpatan":
        x, y = val(*a), val(*b)
        pi = "4*a(1)"
        return run_bc(f"a({B}/{A})" + ("" if x > 0 else f"+({pi})" if y >= 0 else f"-({pi})"))
    if op == "f2xm1": return run_bc(f"e({A}*l(2))-1")
    if op == "fyl2x": return run_bc(f"{B}*l({A})/l(2)")
    if op == "fyl2xp1": return run_bc(f"{B}*l({A}+1)/l(2)")
    return None

stats = {}
for line in open(sys.argv[1]):
    if not line.startswith("DUMP"):
        continue
    f = line.split()
    op = f[1]
    am, ase, bm, bse, hm, hse, mm, mse = (int(x, 16) for x in f[2:])
    r = ref(op, (am, ase), (bm, bse))
    if r is None:
        continue
    want = round64(r)
    s = stats.setdefault(op, [0, 0, 0, 0])
    s[0] += 1
    s[1] += (hm, hse) == want
    s[2] += (mm, mse) == want
    s[3] += (hm, hse) != want and (mm, mse) != want
for op, s in stats.items():
    print(f"{op:8s} differing samples {s[0]:4d}: host correct {s[1]:4d}, ours correct {s[2]:4d}, neither {s[3]:4d}")
