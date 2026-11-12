#!/usr/bin/env python3
"""Static checks of .github/workflows (PyYAML): python3 -I tests/ci/check_workflows.py [workflows dir]

 1. every file parses;
 2. a workflow-level `concurrency` group must depend on `github.run_id` or on the commit message gate: a push without [build]
    skips every job, but its run would still take the group, cancel a running [build] run (cancel-in-progress) or replace
    the pending one (only one run can wait in a group);
 3. a workflow that publishes a release (softprops/action-gh-release) has a concurrency group, so two runs do not race for
    the same tag;
 4. no `${{ inputs.* }}` or `${{ github.event.* }}` text inside a `run:` script (shell injection through a dispatch input
    or a commit message): they go through `env:`.
"""
import glob, os, re, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".github", "workflows")
bad = 0


def fail(f, msg):
    global bad
    bad += 1
    print(f"FAIL {os.path.basename(f)}: {msg}")


INJECT = re.compile(r"\$\{\{\s*(inputs\.|github\.event\.|github\.head_ref)")
for f in sorted(glob.glob(os.path.join(DIR, "*.yml"))):
    try:
        wf = yaml.safe_load(open(f))
    except yaml.YAMLError as e:
        fail(f, f"does not parse: {e}")
        continue
    conc = wf.get("concurrency")
    if conc is not None:
        group = conc.get("group", "") if isinstance(conc, dict) else str(conc)
        if "github.run_id" not in group and "head_commit.message" not in group:
            fail(f, f"workflow-level concurrency group {group!r} is shared with the runs whose jobs are all skipped")
    jobs = wf.get("jobs", {})
    releases = any("action-gh-release" in str(s.get("uses", "")) for j in jobs.values() for s in j.get("steps", []))
    if releases and conc is None:
        fail(f, "publishes a release without a concurrency group")
    for name, job in jobs.items():
        for step in job.get("steps", []):
            run = step.get("run")
            if run and INJECT.search(run):
                fail(f, f"job {name}: step {step.get('name', '?')!r} puts an input or event field inside the script")
print("check_workflows: ok" if not bad else f"check_workflows: {bad} problems")
sys.exit(1 if bad else 0)
