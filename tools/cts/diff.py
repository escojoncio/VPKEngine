#!/usr/bin/env python3
"""Compares two summary.json (summarize.py): regressions, fixes, known failures.

  diff.py OLD.json NEW.json [--known known-fails.txt] [--out diff.md] [--fail-on-regression]

known-fails.txt: one case name or fnmatch pattern per line ('#' comments). A bad result matching it is "known", not a
regression; a known-fail that now passes is listed as stale. A regression is a case that passed before (or is new) and
is now Fail/Crash/Timeout/InternalError/...; a case that is NotRun now is not a regression (the shard did not reach it).
"""
import argparse, fnmatch, json, sys

OK = {"Pass", "NotSupported", "QualityWarning", "CompatibilityWarning", "Waiver"}


def load(p):
    return json.load(open(p))["cases"]


def known_patterns(p):
    if not p:
        return []
    try:
        with open(p) as fh:
            return [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    except OSError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--known")
    ap.add_argument("--out", default="diff.md")
    ap.add_argument("--fail-on-regression", action="store_true")
    a = ap.parse_args()
    try:
        old = load(a.old)
    except (OSError, ValueError):
        old = None
    new = load(a.new)
    pats = known_patterns(a.known)
    known = lambda n: any(fnmatch.fnmatchcase(n, p) for p in pats)
    reg, fix, known_bad, stale, newcases_bad = [], [], [], [], []
    for n, st in sorted(new.items()):
        bad = st not in OK and st != "NotRun"
        if bad and known(n):
            known_bad.append((n, st))
            continue
        if known(n) and st == "Pass":
            stale.append(n)
        o = old.get(n) if old is not None else None
        if bad:
            if o is None:
                if old is not None:
                    newcases_bad.append((n, st))
            elif o in OK:
                reg.append((n, o, st))
        elif st == "Pass" and o is not None and o not in OK and o != "NotRun":
            fix.append((n, o))
    md = ["## Diff against the previous run", ""]
    if old is None:
        md.append("No previous summary to compare with.")
    md.append(f"regressions: {len(reg)}, fixes: {len(fix)}, new cases failing: {len(newcases_bad)}, "
              f"known failures: {len(known_bad)}, stale known-fails (now Pass): {len(stale)}")

    def section(title, rows, fmt):
        if rows:
            md.extend(["", f"### {title} ({len(rows)})", ""] + [fmt(r) for r in rows[:200]])
            if len(rows) > 200:
                md.append(f"... and {len(rows) - 200} more")
    section("Regressions", reg, lambda r: f"- {r[0]}: {r[1]} -> {r[2]}")
    section("Fixes", fix, lambda r: f"- {r[0]}: {r[1]} -> Pass")
    section("New cases failing", newcases_bad, lambda r: f"- {r[0]}: {r[1]}")
    section("Stale known-fails", stale, lambda r: f"- {r}")
    text = "\n".join(md) + "\n"
    open(a.out, "w").write(text)
    print(text)
    return 1 if (a.fail_on_regression and reg) else 0


if __name__ == "__main__":
    sys.exit(main())
