#!/usr/bin/env python3
"""Summarizes Vulkan CTS results (.qpa logs) per group.

  summarize.py --out DIR [--qpa FILE...] [--extra FILE.json...] [--caselist FILE...] [--merge summary.json...] [--title T]

Reads the qpa logs (and/or earlier summary.json files to merge, e.g. one per shard), adds the cases the runner recorded
itself (--extra: {case: [status, message]}) and the cases of the case lists that have no result (NotRun), and writes
DIR/summary.json ({"cases": {name: status}, "msgs": {name: first line of the result text}}) and DIR/summary.md
(counts per top group, worst second-level groups, most frequent failure messages).
"""
import argparse, json, os, re, sys

STATUSES = ["Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning", "Crash", "Timeout",
            "InternalError", "NotRun"]
BAD = {"Fail", "Crash", "Timeout", "InternalError", "ResourceError", "DeviceLost"}
RESULT_RE = re.compile(r'<Result StatusCode="([A-Za-z]+)"\s*(?:/>|>(.*?)(?:</Result>|$))')


def parse_qpa(path):
    """-> (results {case: (status, msg)}, open_case or None, terminated {case: status}). A case still open at the end of
    the file is the one the process died in (killed, or crashed before the crash handler wrote anything)."""
    res, cur, status, msg, term = {}, None, None, "", {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#beginTestCaseResult "):
                cur, status, msg = line[len("#beginTestCaseResult "):].strip(), None, ""
            elif line.startswith("#endTestCaseResult"):
                if cur:
                    res[cur] = (status or "InternalError", msg)
                cur = None
            elif line.startswith("#terminateTestCaseResult"):
                if cur:
                    st = line.split(None, 1)[1].strip() if " " in line.strip() else "Crash"
                    res[cur] = (st, "terminated by the harness (" + st + ")")
                    term[cur] = st
                cur = None
            elif cur is not None and status is None:
                m = RESULT_RE.search(line)
                if m:
                    status = m.group(1)
                    msg = (m.group(2) or "").strip()[:300]
    return res, cur, term


TAG_RE = re.compile(r"<[^>]+>")


def fail_details(path, limit=2500):
    """-> {case: text}: for every case that ended in Fail/InternalError/Crash/Timeout, the log lines (<Text>, <Message>,
    <Result>, tags stripped) the harness wrote, cut to `limit` characters. The summary keeps only the first line."""
    out, cur, buf = {}, None, []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#beginTestCaseResult "):
                cur, buf = line[len("#beginTestCaseResult "):].strip(), []
            elif line.startswith("#endTestCaseResult"):
                txt = "\n".join(buf)
                if cur and re.search(r'<Result StatusCode="(Fail|InternalError|Crash|Timeout|ResourceError)"', txt):
                    out[cur] = txt[:limit]
                cur = None
            elif cur is not None and ("<Text>" in line or "<Message>" in line or "<Result " in line):
                t = TAG_RE.sub("", line).strip()
                buf.append(line.strip() if "<Result " in line else t)
    return out


def fail_full(path, head=7000, tail=14000):
    """-> {case: text}: the complete log of every case that ended in Fail/InternalError/Crash/Timeout, tags kept, head and
    tail kept (focused runs, where the failure message alone does not say what went wrong)."""
    out, cur, buf = {}, None, []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#beginTestCaseResult "):
                cur, buf = line[len("#beginTestCaseResult "):].strip(), []
            elif line.startswith("#endTestCaseResult"):
                txt = "".join(buf)
                if cur and re.search(r'<Result StatusCode="(Fail|InternalError|Crash|Timeout|ResourceError)"', txt):
                    out[cur] = txt if len(txt) <= head + tail else txt[:head] + "\n[...]\n" + txt[-tail:]
                cur = None
            elif cur is not None:
                buf.append(line)
    return out


def read_list(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip() and not l.startswith("#")]


def group_of(case, depth):
    parts = case.split(".")
    return ".".join(parts[1:1 + depth]) if parts[0] == "dEQP-VK" else ".".join(parts[:depth])


def normalize_msg(m):
    m = re.sub(r"0x[0-9a-fA-F]+", "0x#", m)
    return re.sub(r"\d+", "#", m)[:160]


def write_reports(cases, msgs, out, title="Vulkan CTS"):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({"cases": cases, "msgs": msgs}, fh, separators=(",", ":"), sort_keys=True)
    cols = STATUSES + ["Other"]

    def bucket(st):
        return st if st in STATUSES else "Other"

    def table(groups):
        lines = ["| group | total | " + " | ".join(cols) + " | pass % |", "|---|---:|" + "---:|" * len(cols) + "---:|"]
        for g, c in groups:
            tot = sum(c.values())
            denom = tot - c.get("NotSupported", 0) - c.get("NotRun", 0)
            pct = f"{100.0 * c.get('Pass', 0) / denom:.1f}" if denom > 0 else "-"
            lines.append(f"| {g} | {tot} | " + " | ".join(str(c.get(k, 0)) for k in cols) + f" | {pct} |")
        return lines

    def count_by(depth):
        by = {}
        for n, st in cases.items():
            by.setdefault(group_of(n, depth), {})
            b = by[group_of(n, depth)]
            b[bucket(st)] = b.get(bucket(st), 0) + 1
        return by

    top = count_by(1)
    total = {}
    for c in top.values():
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    md = [f"## {title}", ""]
    md += table([("**all**", total)] + sorted(top.items()))
    second = count_by(2)
    worst = sorted(second.items(), key=lambda kv: -sum(v for k, v in kv[1].items() if k in BAD or k == "Other"))
    worst = [w for w in worst if any(k in BAD or k == "Other" for k in w[1])][:30]
    md += ["", "### Worst groups (Fail + Crash + Timeout + InternalError + other)", ""]
    md += table(worst) if worst else ["none"]
    freq = {}
    for n, st in cases.items():
        if st in BAD and msgs.get(n):
            k = (st, normalize_msg(msgs[n]))
            freq.setdefault(k, [0, n])[0] += 1
    md += ["", "### Most frequent failure messages", "", "| count | status | message | example |", "|---:|---|---|---|"]
    for (st, m), (cnt, ex) in sorted(freq.items(), key=lambda kv: -kv[1][0])[:25]:
        md.append(f"| {cnt} | {st} | `{m.replace('|', '/')}` | {ex} |")
    ns = {}
    for n, st in cases.items():
        if st == "NotSupported":
            k = normalize_msg(msgs.get(n, "(no message)"))
            ns.setdefault(k, [0, n])[0] += 1
    md += ["", "### Most frequent NotSupported reasons", "", "| count | message | example |", "|---:|---|---|"]
    for m, (cnt, ex) in sorted(ns.items(), key=lambda kv: -kv[1][0])[:40]:
        md.append(f"| {cnt} | `{m.replace('|', '/')}` | {ex} |")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--qpa", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--caselist", nargs="*", default=[])
    ap.add_argument("--merge", nargs="*", default=[])
    ap.add_argument("--title", default="Vulkan CTS")
    a = ap.parse_args()
    cases, msgs = {}, {}
    for p in a.merge:
        d = json.load(open(p))
        cases.update(d["cases"])
        msgs.update(d.get("msgs", {}))
    for p in a.qpa:
        res, _, _ = parse_qpa(p)
        for n, (st, m) in res.items():
            cases[n] = st
            if m and st != "Pass":
                msgs[n] = m
    for p in a.extra:
        for n, (st, m) in json.load(open(p)).items():
            cases[n] = st
            if m and st != "Pass":
                msgs[n] = m
    for p in a.caselist:
        for n in read_list(p):
            cases.setdefault(n, "NotRun")
    total = write_reports(cases, msgs, a.out, a.title)
    print(json.dumps(total, sort_keys=True))


if __name__ == "__main__":
    main()
