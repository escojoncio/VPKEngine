#!/usr/bin/env python3
"""Tests of tests/ci/check_workflows.py check 6 on throw-away git repositories: python3 -I tests/ci/test_check_workflows.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "check_workflows.py")


def git(root, *a):
    subprocess.run(["git", "-C", root, *a], check=True, capture_output=True)


def case(name, run, expect_ok, files=None, extra=None, wd=None, action=False, shell=None, links=None, nogit=False):
    """files: {path: (content, executable)}, tracked; extra: untracked/ignored files {path: content}."""
    with tempfile.TemporaryDirectory() as root:
        if not nogit: git(root, "init", "-q")
        os.makedirs(os.path.join(root, ".github", "workflows"))
        step = "      - run: |\n" + "".join("          " + l + "\n" for l in run.splitlines())
        if shell: step = f"      - shell: {shell}\n        run: |\n" + "".join("          " + l + "\n" for l in run.splitlines())
        if wd: step = f"      - working-directory: {wd}\n        run: |\n" + "".join("          " + l + "\n" for l in run.splitlines())
        open(os.path.join(root, ".github", "workflows", "w.yml"), "w").write("on: push\njobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n" + step)
        if action:
            os.makedirs(os.path.join(root, ".github", "actions", "a"))
            open(os.path.join(root, ".github", "actions", "a", "action.yml"), "w").write(
                "runs:\n  using: composite\n  steps:\n    - shell: bash\n      run: |\n" + "".join("        " + l + "\n" for l in run.splitlines()))
            open(os.path.join(root, ".github", "workflows", "w.yml"), "w").write("on: push\njobs:\n  j:\n    runs-on: x\n    steps:\n      - run: echo\n")
        open(os.path.join(root, ".gitignore"), "w").write("build/\n")
        for path, (content, ex) in (files or {}).items():
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w").write(content)
            os.chmod(full, 0o755 if ex else 0o644)
            if nogit: continue
            git(root, "add", "--", path)
            git(root, "update-index", "--chmod=" + ("+x" if ex else "-x"), "--", path)
        for path, target in (links or {}).items():  # tracked symlinks
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            os.symlink(target, full)
            if not nogit: git(root, "add", "--", path)
        for path, content in (extra or {}).items():
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w").write(content)
            os.chmod(full, 0o755)
        r = subprocess.run([sys.executable, "-I", CHECK, os.path.join(root, ".github", "workflows"), root], capture_output=True, text=True)
    ok = (r.returncode == 0) == expect_ok
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "\n" + r.stdout + r.stderr))
    return 0 if ok else 1


G = {"tests/a/run.sh": ("#!/bin/sh\n", True), "tests/a/bad.sh": ("#!/bin/sh\n", False)}
bad = 0
bad += case("no build output", "tests/a/run.sh", True, G)
bad += case("untracked ignored build output", "./build/vpaot/vpaot x\nbuild/loader_self/loader_self y", True, G, {"build/vpaot/vpaot": "", "build/loader_self/loader_self": ""})
bad += case("tracked 100644", "tests/a/bad.sh", False, G)
bad += case("sh prefix is fine", "sh tests/a/bad.sh\npython3 tests/a/bad.sh", True, G)
bad += case("missing script", "tests/a/nope.sh", False, G)
bad += case("cd && ./x", "cd tests/a && ./bad.sh", False, G)
bad += case("cd && ./x ok", "cd tests/a && ./run.sh", True, G)
bad += case("working-directory", "./bad.sh", False, G, wd="tests/a")
bad += case("working-directory ok", "./run.sh", True, G, wd="tests/a")
for form in ["true && tests/a/bad.sh", "set -e; tests/a/bad.sh", "tests/a/bad.sh; echo done", "echo | tests/a/bad.sh", "VAR=1 tests/a/bad.sh",
             "time tests/a/bad.sh", "env X=1 tests/a/bad.sh", "sudo tests/a/bad.sh", "exec tests/a/bad.sh", "! tests/a/bad.sh",
             "ls | xargs tests/a/bad.sh", "bash -c 'tests/a/bad.sh'", "$GITHUB_WORKSPACE/tests/a/bad.sh", "${{ github.workspace }}/tests/a/bad.sh",
             "for i in 1 2; do tests/a/bad.sh; done", "{ tests/a/bad.sh; }", "(tests/a/bad.sh)", "if ! tests/a/bad.sh; then echo; fi",
             "true \\\n  && tests/a/bad.sh", "(cd tests/a && ./bad.sh)", "tests/a/bad.sh > out.log 2>&1", "cat <<EOF\nx\nEOF\ntests/a/bad.sh"]:
    bad += case("detects: " + form.replace("\n", " NL "), form, False, G)
for form in ["cat <<EOF\ntests/a/bad.sh\nEOF", "cat <<'EOF' > f\ntests/a/bad.sh\nEOF\necho", "echo 'tests/a/bad.sh'", "x=tests/a/bad.sh", "echo tests/a/bad.sh",
             "# tests/a/bad.sh", "(cd tests/a); echo ok",
             "X=(runtime/x.c tests/a/bad.sh)\necho ${X[@]}", "X+=(tests/a/bad.sh)", "X=(\n  tests/a/bad.sh\n  tests/a/bad.sh\n)\nfor f in \"${X[@]}\"; do echo $f; done",
             "case \"$x\" in\n  tests/*) echo a;;\n  (tests/a/bad.sh|runtime/x.c) echo b;;\n  *) echo c;;\nesac",
             "case $x in tests/a/bad.sh) echo;; esac", "cat <<\\EOF\ntests/a/bad.sh\nEOF"]:
    bad += case("no false positive: " + form.replace("\n", " NL "), form, True, G)
bad += case("case body is a command", "case $x in\n  a) tests/a/bad.sh;;\nesac", False, G)
bad += case("case after a case", "case $x in a) echo;; esac\ntests/a/bad.sh", False, G)
bad += case("/bin/bash -c with a path", "/bin/bash -c 'tests/a/bad.sh'", False, G)
bad += case("pwsh step is skipped", "tests/a/bad.sh", True, G, shell="pwsh")
bad += case("python step is skipped", "tests/a/bad.sh", True, G, shell="python")
bad += case("bash step is checked", "tests/a/bad.sh", False, G, shell="bash")
bad += case("tracked symlink is not judged", "tests/a/link.sh", True, G, links={"tests/a/link.sh": "run.sh"})
bad += case("git failing is a failure", "tests/a/run.sh", False, G, nogit=True)
bad += case("no path, no git needed", "echo hi", True, G, nogit=True)
bad += case("composite action", "tests/a/bad.sh", False, G, action=True)
print("test_check_workflows: " + ("ok" if not bad else f"{bad} failed"))
sys.exit(1 if bad else 0)
