#!/usr/bin/env python3
"""Runs one lot (case list) of the Vulkan CTS with deqp-vk, surviving crashes and hangs.

  run_lot.py --deqp PATH [--lib libVPVulkan.dylib] --caselist FILE --out DIR [--budget-min 50] [--idle-s 300] [--env K=V]...

deqp-vk runs the whole list in one process. When the process dies (crash, abort, kill), the case it was running is
recorded as Crash (or Timeout when the log stopped growing for --idle-s seconds and the runner killed it), and deqp-vk is
started again with the cases that have no result yet, skipping the one that died (--deqp-caselist-file resume). When the
time budget (--budget-min) ends, the rest of the list is reported as NotRun.

Writes DIR/attempt-N.qpa, DIR/attempt-N.log (stdout/stderr), DIR/summary.json + summary.md (summarize.py) and
DIR/lot.json (attempts, exit codes, reasons) and
DIR/fails.txt (log text of the failed cases). The exit status is 0 unless deqp-vk could not start at all (2).
Extra arguments after `--` go to deqp-vk. Without --lib no --deqp-vk-library-path is given (the GLES modules, deqp-gles*,
reach the driver through the EGL shim; --env sets the variables of the child, e.g. VPGLES_VULKAN_LIB).
"""
import argparse, json, os, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402


def run_once(cmd, cwd, qpa, log, idle_s, deadline, env=None):
    """-> (exit code, why) with why in {"exit", "idle", "budget"}."""
    with open(log, "wb") as lf:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        last_size, last_change = -1, time.time()
        while True:
            try:
                return p.wait(timeout=5), "exit"
            except subprocess.TimeoutExpired:
                pass
            size = os.path.getsize(qpa) if os.path.exists(qpa) else 0
            now = time.time()
            if size != last_size:
                last_size, last_change = size, now
            why = "idle" if now - last_change > idle_s else "budget" if now > deadline else None
            if why:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                return -signal.SIGKILL, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deqp", required=True)
    ap.add_argument("--lib")
    ap.add_argument("--env", action="append", default=[], help="K=V for the deqp process")
    ap.add_argument("--caselist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd", help="directory deqp-vk runs in (its data dir); default: the binary's directory")
    ap.add_argument("--budget-min", type=float, default=50)
    ap.add_argument("--idle-s", type=float, default=300)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--full", action="store_true",
                    help="focused run: shader sources in the log, fails-full.txt with the whole log of every failed case")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    deqp = os.path.abspath(a.deqp)
    cwd = os.path.abspath(a.cwd) if a.cwd else os.path.dirname(deqp)
    cases = summarize.read_list(a.caselist)
    todo = list(cases)
    deadline = time.time() + a.budget_min * 60
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env)) if a.env else None
    extra, lot, qpas, startup_failed = {}, [], [], False
    done = set()
    for n in range(1, a.max_attempts + 1):
        if not todo or time.time() > deadline:
            break
        lst = os.path.join(out, f"remaining-{n}.txt")
        with open(lst, "w") as fh:
            fh.write("\n".join(todo) + "\n")
        qpa, log = os.path.join(out, f"attempt-{n}.qpa"), os.path.join(out, f"attempt-{n}.log")
        cmd = [deqp] + ([f"--deqp-vk-library-path={os.path.abspath(a.lib)}"] if a.lib else []) + [
               f"--deqp-caselist-file={lst}",
               f"--deqp-log-filename={qpa}", "--deqp-log-images=disable",
               "--deqp-log-shader-sources=" + ("enable" if a.full else "disable"),
               "--deqp-crashhandler=enable", "--deqp-watchdog=enable", "--deqp-shadercache=disable"] + a.extra
        t0 = time.time()
        rc, why = run_once(cmd, cwd, qpa, log, a.idle_s, deadline, env)
        res, open_case, term = summarize.parse_qpa(qpa) if os.path.exists(qpa) else ({}, None, {})
        progress = len(res)
        for c, (st, m) in res.items():
            done.add(c)
        if open_case:                       # the case the process died in
            if why == "idle":
                extra[open_case] = ["Timeout", f"no output for {int(a.idle_s)} s, killed by the runner"]
            elif why == "budget":
                extra[open_case] = ["NotRun", "time budget ended while it ran"]
            else:
                extra[open_case] = ["Crash", f"deqp-vk exited with code {rc} inside the case"]
            done.add(open_case)
            progress += 1
        lot.append({"attempt": n, "exit": rc, "why": why, "cases_done": progress, "open_case": open_case,
                    "seconds": round(time.time() - t0, 1), "todo": len(todo)})
        print(f"attempt {n}: exit={rc} why={why} done={progress} open={open_case} remaining={len(todo) - progress}", flush=True)
        qpas.append(qpa)
        todo = [c for c in todo if c not in done]
        if progress == 0 and rc != 0:       # deqp-vk could not even start the first case
            startup_failed = True
            tail = open(log, errors="replace").read()[-1500:] if os.path.exists(log) else ""
            print("deqp-vk made no progress; giving up. Log tail:\n" + tail, flush=True)
            break
        if rc == 0 and not open_case and not term:
            # Clean exit but cases missing (names unknown to this build): leave them NotRun.
            break
    extra_path = os.path.join(out, "extra.json")
    json.dump(extra, open(extra_path, "w"))
    json.dump({"attempts": lot, "cases": len(cases), "left": len(todo), "startup_failed": startup_failed},
              open(os.path.join(out, "lot.json"), "w"), indent=1)
    with open(os.path.join(out, "fails.txt"), "w") as fh:     # log text of every failed case (the summary has 1 line)
        for q in qpas:
            if os.path.exists(q):
                for c, t in sorted(summarize.fail_details(q).items()):
                    fh.write(f"=== {c}\n{t}\n")
    if a.full:
        with open(os.path.join(out, "fails-full.txt"), "w") as fh:
            for q in qpas:
                if os.path.exists(q):
                    for c, t in sorted(summarize.fail_full(q).items()):
                        fh.write(f"=== {c}\n{t}\n")
    sargs = ["--out", out, "--qpa", *qpas, "--extra", extra_path, "--caselist", a.caselist, "--title",
             f"Lot {os.path.basename(a.caselist)}"]
    sys.argv = ["summarize.py"] + sargs
    summarize.main()
    return 2 if startup_failed else 0


if __name__ == "__main__":
    sys.exit(main())
