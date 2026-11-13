#!/usr/bin/env python3
"""Tests of tests/ci/check_workflows.py check 6 on throw-away git repositories: python3 -I tests/ci/test_check_workflows.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "check_workflows.py")


def git(root, *a):
    subprocess.run(["git", "-C", root, *a], check=True, capture_output=True)


_repos = {}  # one throw-away repository per set of files (git init and the adds are the slow part)
_jobs = []


def repo_for(key, files, extra, links, nogit, action_run):
    if key in _repos: return _repos[key]
    root = tempfile.mkdtemp(prefix="cwtest")
    if not nogit: git(root, "init", "-q")
    if action_run is not None:
        os.makedirs(os.path.join(root, ".github", "actions", "a"))
        open(os.path.join(root, ".github", "actions", "a", "action.yml"), "w").write(
            "runs:\n  using: composite\n  steps:\n    - shell: bash\n      run: |\n" + "".join("        " + l + "\n" for l in action_run.splitlines()))
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
    _repos[key] = root
    return root


class Result:
    def __init__(self, returncode, stdout, stderr): self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_workflows_under_test", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _module()


def run_check(root, yml, as_dir=False):
    """Runs the checker's main() in this process (starting python and importing yaml for each of the ~200 cases, and the 10000 of the
    fuzz, would cost most of the time); an exception of the checker is reported as a Traceback on stderr, like the real one.
    yml: text or bytes; as_dir: w.yml is a directory."""
    with tempfile.TemporaryDirectory() as wfdir:
        if as_dir: os.mkdir(os.path.join(wfdir, "w.yml"))
        else: open(os.path.join(wfdir, "w.yml"), "wb").write(yml if isinstance(yml, bytes) else yml.encode())
        return run_check_in(root, wfdir)


def run_check_in(root, wfdir):
    import contextlib, io, traceback
    out, err = io.StringIO(), io.StringIO()
    code = 0
    if True:
        # (a temp dir inside a git work tree must not find that repository: GIT_CEILING_DIRECTORIES stops the search above it)
        saved = {k: os.environ.get(k) for k in ("GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE")}
        os.environ["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.realpath(root))
        os.environ.pop("GIT_DIR", None); os.environ.pop("GIT_WORK_TREE", None)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try: code = MOD.main(["check_workflows.py", wfdir, root])
                except BaseException: err.write(traceback.format_exc()); code = 1
        finally:
            for k, v in saved.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
    return Result(code, out.getvalue(), err.getvalue())


def case(name, run, expect_ok, files=None, extra=None, wd=None, action=False, shell=None, links=None, nogit=False, runs_on="ubuntu-latest", env_yaml=""):
    """files: {path: (content, executable)}, tracked; extra: untracked/ignored files {path: content}. Queued; run_all() runs them."""
    lines = "".join("          " + l + "\n" for l in run.splitlines())
    head = f"      - shell: {shell}\n        run: |\n" if shell else (f"      - working-directory: {wd}\n        run: |\n" if wd else "      - run: |\n")
    yml = "on: push\n" + env_yaml + "jobs:\n  j:\n    runs-on: " + runs_on + "\n    steps:\n" + head + lines
    if action: yml = "on: push\njobs:\n  j:\n    runs-on: x\n    steps:\n      - run: echo\n"
    key = (repr(sorted((files or {}).items())), repr(sorted((extra or {}).items())), repr(sorted((links or {}).items())), nogit, run if action else None)
    _jobs.append((name, expect_ok, key, (files, extra, links, nogit, run if action else None), (yml, False), None))
    return 0


def yml_case(name, yml, expect_ok, files=None, text=None, as_dir=False):
    """A workflow file given as it is: the checker must answer (never a traceback), with `text` in its output if given."""
    key = (repr(sorted((files or {}).items())), "[]", "[]", False, None)
    _jobs.append((name, expect_ok, key, (files, None, None, False, None), (yml, as_dir), text))
    return 0


def run_all():
    for _, _, key, args, _, _ in _jobs: repo_for(key, *args)
    results = [run_check(_repos[j[2]], *j[4]) for j in _jobs]
    failed = 0
    for (name, expect_ok, _, _, _, text), r in zip(_jobs, results):
        ok = (r.returncode == 0) == expect_ok and "Traceback" not in r.stderr and (text is None or text in r.stdout + r.stderr)
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else "\n" + r.stdout + r.stderr))
        failed += 0 if ok else 1
    return failed


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
more = [  # (name, script) the checker must flag
    ("time -p", "time -p tests/a/bad.sh"), ("function f {", "function f { tests/a/bad.sh; }\nf"), ("f() {", "f() { tests/a/bad.sh; }"),
    ("coproc", "coproc tests/a/bad.sh"), ("coproc NAME {", "coproc NAME { tests/a/bad.sh; }"),
    ("case ;&", "case $x in a) echo;& b) tests/a/bad.sh;; esac"), ("case ;;&", "case $x in a) echo;;& b) tests/a/bad.sh;; esac"),
    ("$(...)", "x=$(tests/a/bad.sh)"), ("backticks", "x=`tests/a/bad.sh`"), ('"$(...)"', 'echo "$(tests/a/bad.sh)"'),
    ("eval", "eval 'tests/a/bad.sh'"), ("find -exec", "find . -name x -exec tests/a/bad.sh {} \\;"),
    ("xargs -I{}", "ls | xargs -I{} tests/a/bad.sh {}"), ("xargs -I {}", "ls | xargs -I {} tests/a/bad.sh {}"),
    ("timeout N", "timeout 10 tests/a/bad.sh"), ("timeout -s KILL N", "timeout -s KILL 10 tests/a/bad.sh"),
    ("sudo -u", "sudo -u nobody tests/a/bad.sh"), ("nice -n", "nice -n 5 tests/a/bad.sh"), ("strace -o", "strace -f -o /tmp/t tests/a/bad.sh"),
    ("ccache", "ccache tests/a/bad.sh"), ("env -u", "env -u X Y=1 tests/a/bad.sh"), ("pushd", "pushd tests/a\n./bad.sh"),
    ("pushd popd", "pushd tests/a\npopd\ntests/a/bad.sh"), ("variable", "S=tests/a\n$S/bad.sh"), ("variable braces+export", "export S=tests/a\n${S}/bad.sh"),
    ("balanced quotes in $()", "x=$(echo ')'; tests/a/bad.sh)"), ("balanced ${x:-\"}\"}", 'echo ${x:-"}"}; tests/a/bad.sh'), ("ANSI-C", "echo $'it\\'s ) x'; tests/a/bad.sh"),
]
more += [
    ("comment with a quote inside $()", "x=$(foo # it's a comment\n)\ntests/a/bad.sh"), ("single quote inside \"${x:-'}\"", "echo \"${x:-'}\"; tests/a/bad.sh"),
    ("heredoc piped to bash", "cat <<\"EOF\" | bash\ntests/a/bad.sh\nEOF"), ("bash <<EOF", "bash <<EOF\ntests/a/bad.sh\nEOF"),
    ("bash -s <<'EOF'", "bash -s <<'EOF'\ncd tests/a && ./bad.sh\nEOF"), ("command X", "command tests/a/bad.sh"),
    ("env -C DIR", "env -C tests/a ./bad.sh"), ("env --chdir=DIR", "env --chdir=tests/a ./bad.sh"), ("unclosed $(", "x=$(tests/a/bad.sh"),
    ("env --chdir DIR", "env --chdir tests/a ./bad.sh"), ("env --unset VAR", "env --unset X tests/a/bad.sh"), ("env -u VAR", "env -u X tests/a/bad.sh"),
    ("env -S", "env -S 'tests/a/bad.sh arg'"), ("env --split-string", "env --split-string='tests/a/bad.sh arg'"), ("env -S separate", "env --split-string 'tests/a/bad.sh arg'"),
    ("env -CDIR", "env -Ctests/a ./bad.sh"), ("command -p X", "command -p tests/a/bad.sh"),
    ("heredoc through tee to bash", "cat <<EOF | tee log | bash\ntests/a/bad.sh\nEOF"), ("here-string to bash", 'bash <<< "tests/a/bad.sh"'),
    ("here-string through cat", "cat <<< 'tests/a/bad.sh' | bash"), ("here-string to sh -s", "sh -s <<< tests/a/bad.sh"),
    ("xargs -I{} sh -c '{}'", "xargs -I{} sh -c '{}' <<< tests/a/bad.sh"), ("xargs -I X bash -c with a heredoc", "xargs -I X bash -c 'X arg' <<EOF\ntests/a/bad.sh\nEOF"),
    ("xargs -i sh -c through cat", "cat <<< tests/a/bad.sh | xargs -i sh -c '{}'"), ("xargs --replace sh -c", "xargs --replace=@ sh -c '@' <<< tests/a/bad.sh"),
    ("xargs -I{} sudo sh -c", "xargs -I{} sudo sh -c '{}' <<< tests/a/bad.sh"),
    ("heredoc with the pipe at the end of its line", "cat <<EOF |\ntests/a/bad.sh\nEOF\nbash"), ("here-string to cat, pipe on the next line", "cat <<< tests/a/bad.sh |\n  bash"),
    ("cat >&1", "cat <<EOF >&1 | bash\ntests/a/bad.sh\nEOF"), ("cat 1>&1", "cat <<EOF 1>&1 | bash\ntests/a/bad.sh\nEOF"),
    ("bash -o pipefail then stdin script", "bash -o pipefail <<EOF\ntests/a/bad.sh\nEOF"), ("bash -eo pipefail", "bash -eo pipefail <<EOF\ntests/a/bad.sh\nEOF"),
    ("bash --rcfile x stdin", "bash --rcfile x <<EOF\ntests/a/bad.sh\nEOF"), ("bash -s -- x", "bash -s -- tests/a/run.sh <<EOF\ntests/a/bad.sh\nEOF"),
    ("sudo bash -s", "sudo bash -s <<< tests/a/bad.sh"), ("bash -O extglob -s", "bash -O extglob -s <<< tests/a/bad.sh"),
    ("bash -e -c", "bash -e -c 'tests/a/bad.sh'"), ("bash -ec", "bash -ec 'tests/a/bad.sh'"), ("bash -o pipefail -c", "bash -o pipefail -c 'tests/a/bad.sh'"),
    ("elif (( 1 << 2 ))", "if false; then :; elif (( 1 << 2 )); then :; fi\ntests/a/bad.sh"), ("! (( 1 << 2 ))", "! (( 1 << 2 ))\ntests/a/bad.sh"),
    ("time (( 1 << 2 ))", "time (( 1 << 2 ))\ntests/a/bad.sh"),
    ("bash - <<EOF", "bash - <<EOF\ntests/a/bad.sh\nEOF"), ("bash /dev/stdin <<EOF", "bash /dev/stdin <<EOF\ntests/a/bad.sh\nEOF"),
    ("bash -s -- arg <<EOF", "bash -s -- x <<EOF\ntests/a/bad.sh\nEOF"), ("bash -s with a script-looking operand", "bash -s tests/a/run.sh <<EOF\ntests/a/bad.sh\nEOF"),
    ("cat <x <<EOF: the heredoc wins", "cat <x <<EOF | bash\ntests/a/bad.sh\nEOF"), ("xargs -ri sh -c", "xargs -ri sh -c '{}' <<< tests/a/bad.sh"),
    ("xargs -r -I{} sh -c", "xargs -r -I{} sh -c '{}' <<< tests/a/bad.sh"), ("timeout --signal KILL 5 X", "timeout --signal KILL 5 tests/a/bad.sh"),
    ("timeout --kill-after N", "timeout --kill-after 3 5 tests/a/bad.sh"), ("/usr/bin/env bash", "/usr/bin/env bash <<EOF\ntests/a/bad.sh\nEOF"),
    ("/usr/bin/env X", "/usr/bin/env tests/a/bad.sh"), ("sudo --user root X", "sudo --user root tests/a/bad.sh"),
    ("the line after an arithmetic command", "(( x = 1 << 3 ))\ntests/a/bad.sh"), ("for (( ... << ... ))", "for (( i = 1 << 0; i < 3; i++ )); do :; done\ntests/a/bad.sh"),
    ("heredoc through |& to bash", "cat <<EOF |& bash\ntests/a/bad.sh\nEOF"),
    ("heredoc through tee >(cat) to bash", "cat <<EOF | tee >(cat) | bash\ntests/a/bad.sh\nEOF"), ("command in a process substitution", "tee >(tests/a/bad.sh) <<< x"),
    ("heredoc through cat 2>&1", "cat <<EOF 2>&1 | bash\ntests/a/bad.sh\nEOF"), ("heredoc through cat 2>log", "cat <<EOF 2>log | bash\ntests/a/bad.sh\nEOF"),
    ("heredoc through cat -", "cat - <<EOF | bash\ntests/a/bad.sh\nEOF"), ("heredoc through tee file", "cat <<EOF | tee log.txt | bash\ntests/a/bad.sh\nEOF"), ("env -S bash with a heredoc", "env -S 'bash -e' <<EOF\ntests/a/bad.sh\nEOF"),
    ("env -S bash with a here-string", "env --split-string='bash -e' <<< tests/a/bad.sh"), ("env -iC DIR", "env -iC tests/a ./bad.sh"),
    ("env -iCDIR", "env -iCtests/a ./bad.sh"), ("env -i -C DIR", "env -i -C tests/a ./bad.sh"), ("env -iu VAR", "env -iu X tests/a/bad.sh"),
    ("heredoc through cat and pv to bash", "cat <<EOF | pv | bash\ntests/a/bad.sh\nEOF"),
    ("unclosed dq", 'echo "$(true) x"; tests/a/bad.sh; echo "'),
]
for n_, sc in more:
    bad += case("detects: " + n_, sc, False, G)
for n_, sc in [("command -pv", "command -pv tests/a/bad.sh"), ("command -vp", "command -vp tests/a/bad.sh"), ("env --chdir DIR into a good script", "env --chdir tests/a ./run.sh"),
               ("env --unset consumes its value", "env --unset tests/a/bad.sh true"), ("env -S of a good script", "env -S 'tests/a/run.sh x'"),
               ("here-string to cat is data", "cat <<< 'tests/a/bad.sh'"), ("heredoc through tee to cat", "cat <<EOF | tee log | cat\ntests/a/bad.sh\nEOF"),
               ("heredoc to cat is data", "cat <<EOF | cat\ntests/a/bad.sh\nEOF"), ("command -v", "command -v tests/a/bad.sh"), ("command -V", "command -V tests/a/bad.sh"),
               ("env -C into a good script", "env -C tests/a ./run.sh"), ("env -C consumes its value", "env -C tests/a/bad.sh true"),
               ("heredoc to a file", "cat <<EOF > out.txt\ntests/a/bad.sh\nEOF"), ("popd goes back", "pushd tests/a\npopd\n./run.sh"), ("single quoted $()", "echo '$(tests/a/bad.sh)'"), ("variable to a good script", "S=tests/a\n$S/run.sh"),
               ("find without exec", "find tests/a/bad.sh -name x"), ("grep of a here-string into sh", "grep x <<<tests/a/bad.sh | sh"), ("heredoc through python3 to bash", "cat <<EOF | python3 - | bash\ntests/a/bad.sh\nEOF"),
               ("heredoc through sed to bash", "cat <<EOF | sed s/a/b/ | bash\ntests/a/bad.sh\nEOF"), ("heredoc through jq to sh", "jq . <<< tests/a/bad.sh | sh"),
               ("env -iC DIR into a good script", "env -iC tests/a ./run.sh"), ("env -iC consumes its value", "env -iC tests/a/bad.sh true"),
               ("cat >&2", "cat <<EOF >&2 | bash\ntests/a/bad.sh\nEOF"), ("cat 1>&2", "cat <<EOF 1>&2 | bash\ntests/a/bad.sh\nEOF"),
               ("xargs sudo bash", "cat <<EOF | xargs sudo bash\ntests/a/bad.sh\nEOF"), ("xargs env bash", "cat <<EOF | xargs env bash\ntests/a/bad.sh\nEOF"),
               ("xargs -a FILE does not read the here-string", "xargs -a list.txt -I{} sh -c '{}' <<< tests/a/bad.sh"),
               ("xargs --arg-file", "xargs --arg-file=list.txt -I{} sh -c '{}' <<< tests/a/bad.sh"), ("xargs -I{} {} (replacement never in the name)", "xargs -I{} {} <<< tests/a/bad.sh"),
               ("xargs -I{} sh -c without {} in the script", "xargs -I{} sh -c 'echo x' <<< tests/a/bad.sh"), ("xargs -I{} sh -c of a good script", "xargs -I{} sh -c '{}' <<< tests/a/run.sh"),
               ("xargs without a command only echoes", 'trim=$(xargs <<< "  tests/a/bad.sh ")'), ("xargs echo fed", "xargs echo <<< tests/a/bad.sh"),
               ("xargs sh runs the file with sh", "xargs sh <<< tests/a/bad.sh"), ("xargs -I{} sh -c fed", "xargs -I{} sh -c 'sh {}' <<< tests/a/bad.sh"),
               ("cat of a file does not pass the heredoc", "cat header.sh <<EOF | bash\ntests/a/bad.sh\nEOF"),
               ("cat with output redirected", "cat <<EOF >log | bash\ntests/a/bad.sh\nEOF"), ("cat with appended output", "cat <<EOF >>log | bash\ntests/a/bad.sh\nEOF"),
               ("tee with output redirected", "cat <<EOF | tee >log | bash\ntests/a/bad.sh\nEOF"), ("cat with input redirected after the heredoc", "cat <<EOF <x | bash\ntests/a/bad.sh\nEOF"),
               ("bash SCRIPT <<< list file", "bash tests/a/run.sh <<< tests/a/bad.sh"), ("sh SCRIPT <<< missing", "sh tools/x.sh <<< gpu/missing/file.c"),
               ("heredoc to sh SCRIPT through tee", "cat <<EOF | tee x | sh tests/a/run.sh\ntests/a/bad.sh\nEOF"), ("bash -e SCRIPT heredoc", "bash -e tests/a/run.sh <<EOF\ntests/a/bad.sh\nEOF"),
               ("bash <<EOF <file reads the file", "bash <<EOF <x\ntests/a/bad.sh\nEOF"),
               ("script operand then -s", "bash tests/a/run.sh -s <<EOF\ntests/a/bad.sh\nEOF"), ("script operand then -vs", "bash tests/a/run.sh -vs <<EOF\ntests/a/bad.sh\nEOF"),
               ("script operand then -fast", "bash tests/a/run.sh -fast <<EOF\ntests/a/bad.sh\nEOF"), ("script operand then --fast -s", "sh tests/a/run.sh --fast -s <<< tests/a/bad.sh"),
               ("sudo bash SCRIPT -s", "sudo bash tests/a/run.sh -s <<< tests/a/bad.sh"), ("script operand then -c", "bash tests/a/run.sh -c tests/a/bad.sh"),
               ("-o pipefail then a script operand", "bash -o pipefail tests/a/run.sh -s <<< tests/a/bad.sh"), ("bash <<EOF >log is not followed", "bash <<EOF >log\ntests/a/bad.sh\nEOF"),
               ("cat -n changes the text", "cat -n <<EOF | bash\ntests/a/bad.sh\nEOF"), ("cat -A", "cat -A <<EOF | bash\ntests/a/bad.sh\nEOF"), ("cat -vET", "cat -vET <<EOF | bash\ntests/a/bad.sh\nEOF"),
               ("xargs -a FILE separate", "xargs -a l.txt -I{} sh -c '{}' <<< tests/a/bad.sh"), ("xargs -aFILE attached", "xargs -al.txt -I{} sh -c '{}' <<< tests/a/bad.sh"),
               ("xargs --arg-file FILE separate", "xargs --arg-file l.txt -I{} sh -c '{}' <<< tests/a/bad.sh"),
               ("(( x = 1 << 3 )) is arithmetic", "(( x = 1 << 3 ))\ntests/a/run.sh"),
               ("cat of a process substitution", "cat <(echo) <<EOF | bash\ntests/a/bad.sh\nEOF"),
               ("xargs fed with a good script", "xargs <<< tests/a/run.sh"), ("env -S bash with a good heredoc", "env -S 'bash -e' <<EOF\ntests/a/run.sh\nEOF"),
               ("timeout of a good script", "timeout 5 tests/a/run.sh")]:
    bad += case("no false positive: " + n_, sc, True, G)
bad += case("windows job defaults to pwsh", "tests/a/bad.sh", True, G, runs_on="windows-latest")
bad += case("windows job with shell bash", "tests/a/bad.sh", False, G, runs_on="windows-latest", shell="bash")
bad += case("shell: /usr/bin/bash -e {0}", "tests/a/bad.sh", False, G, shell="'/usr/bin/bash -e {0}'")
bad += case("env: variable", "$S/bad.sh", False, G, env_yaml="env:\n  S: tests/a\n")
bad += case("composite action", "tests/a/bad.sh", False, G, action=True)
DEEP = "jobs:\n  j:\n    steps:\n      - run: |\n"
for n_, y, ok_, t_, d_ in [
    ("concurrency group null", "on: push\nconcurrency:\n  group:\njobs: {}\n", False, "concurrency group", False),
    ("concurrency group 5", "on: push\nconcurrency: {group: 5}\njobs: {}\n", False, "concurrency group", False),
    ("concurrency group true", "on: push\nconcurrency: {group: true}\njobs: {}\n", False, "concurrency group", False),
    ("concurrency group list", "on: push\nconcurrency: {group: [a]}\njobs: {}\n", False, "concurrency group", False),
    ("job concurrency group null in gles-cts", "on: push\njobs:\n  j:\n    concurrency: {group: , cancel-in-progress: true}\n", True, None, False),
    ("NUL in a run", 'on: push\njobs:\n  j:\n    steps:\n      - run: "tests/a/bad.sh\\0 x"\n', True, None, False),
    ("NUL in a working directory", 'on: push\njobs:\n  j:\n    steps:\n      - run: tests/a/bad.sh\n        working-directory: "a\\0"\n', True, None, False),
    ("run: a date", "on: push\njobs:\n  j:\n    steps:\n      - run: 2024-01-01\n", True, None, False),
    ("run: a float", "on: push\njobs:\n  j:\n    steps:\n      - run: 1.5e3\n", True, None, False),
    ("invalid UTF-8", b"on: push\n# \xff\xfe\njobs: {}\n", False, "not UTF-8", False),
    ("a directory called w.yml", "", False, "cannot be read", True),
    ("YAML nested 1500 deep", "a: " + "[" * 1500 + "]" * 1500 + "\n", False, "does not parse", False),
    ("eval x1200", DEEP + "          " + "eval " * 1200 + "tests/a/bad.sh\n", True, "WARN", False),
    ("$( x1100", DEEP + "          echo " + "$(" * 1100 + "true" + ")" * 1100 + "\n", True, None, False),
    ("( x1200", DEEP + "          " + "(" * 1200 + "true" + ")" * 1200 + "\n", True, None, False),
    ("{ x1200", DEEP + "          " + "{ " * 1200 + "true; " + "} " * 1200 + "\n", True, None, False),
]:
    yml_case(n_, y, ok_, G, t_, d_)
for n_, y, ok_, t_ in [
    ("empty file", "", True, None), ("only a comment", "# x\n", True, None), ("a list", "- a\n", False, "not a mapping"), ("a scalar", "x\n", False, "not a mapping"),
    ("jobs: []", "on: push\njobs: []\n", True, None), ("jobs: null", "on: push\njobs:\n", True, None), ("jobs: a string", "on: push\njobs: x\n", False, "jobs is not a mapping"),
    ("j: null", "on: push\njobs:\n  j:\n", True, None), ("j: a list", "on: push\njobs:\n  j: [a]\n", False, "not a mapping"),
    ("steps: null", "on: push\njobs:\n  j:\n    steps:\n", True, None), ("steps: a string", "on: push\njobs:\n  j:\n    steps: x\n", False, "steps is not a list"),
    ("- null step", "on: push\njobs:\n  j:\n    steps:\n      -\n", True, None), ("a step that is a string", "on: push\njobs:\n  j:\n    steps:\n      - x\n", False, "is not a mapping"),
    ("run: 123", "on: push\njobs:\n  j:\n    steps:\n      - run: 123\n", True, None), ("run: [a]", "on: push\njobs:\n  j:\n    steps:\n      - run: [a]\n", False, "not text"),
    ("run: {a: b}", "on: push\njobs:\n  j:\n    steps:\n      - run: {a: b}\n", False, "not text"), ("run: true", "on: push\njobs:\n  j:\n    steps:\n      - run: true\n", True, None),
    ("env: [a]", "on: push\nenv: [a]\njobs:\n  j:\n    steps:\n      - run: x\n", False, "env is not a mapping"),
    ("step env: [a]", "on: push\njobs:\n  j:\n    steps:\n      - run: x\n        env: [a]\n", False, "env is not a mapping"),
    ("defaults: {run: null}", "on: push\ndefaults:\n  run:\njobs:\n  j:\n    defaults:\n      run:\n    steps:\n      - run: x\n", True, None),
    ("defaults: [a]", "on: push\ndefaults: [a]\njobs:\n  j:\n    defaults: x\n    steps:\n      - run: x\n", True, None),
    ("defaults.run: [a]", "on: push\njobs:\n  j:\n    defaults:\n      run: [a]\n    steps:\n      - run: x\n", True, None),
    ("working-directory as a list", "on: push\njobs:\n  j:\n    steps:\n      - run: x\n        working-directory: [a]\n", True, None),
    ("shell as a list", "on: push\njobs:\n  j:\n    steps:\n      - run: x\n        shell: [a]\n", True, None),
    ("concurrency as a list", "on: push\nconcurrency: [a]\njobs:\n  j:\n    steps:\n      - run: x\n", False, "concurrency group"),
    ("a file that does not parse", "a: [\n", False, "does not parse"),
]:
    yml_case(n_, y, ok_, G, t_)
bad += run_all()


def fuzz(n):
    """Structural fuzz: a valid workflow with random parts replaced by values of the wrong type; the checker must never raise."""
    import copy, datetime, random, yaml
    rnd = random.Random(20260101)
    junk = [None, True, False, 0, 5, 1.5, "", "x", "bash", "${{ github.run_id }}", datetime.date(2024, 1, 1), [], [None], [1, "a"], ["a", {"b": None}], {}, {"a": None}, {"run": None},
            {"group": None}, {"group": 5}, {"group": True}, {"group": [1]}, {"run": {"shell": None, "working-directory": None}}, b"\xff", {1, 2}, [[[]]], {"a": {"b": {"c": [None]}}}]

    def base():
        return {"on": "push", "concurrency": {"group": "g-${{ github.run_id }}", "cancel-in-progress": False}, "env": {"A": "1"}, "defaults": {"run": {"shell": "bash", "working-directory": "tests"}},
                "jobs": {"j": {"runs-on": "ubuntu-latest", "env": {"B": "2"}, "defaults": {"run": {"shell": "bash"}}, "concurrency": {"group": "x", "cancel-in-progress": True},
                               "steps": [{"name": "s", "run": "tests/a/run.sh | xargs -I{} sh -c '{}'", "shell": "bash", "working-directory": "tests", "env": {"C": "3"}, "uses": "u"}]}}}

    def pick(d):
        """A random place of the document: (container, key), by walking down from the root."""
        parent, key, cur = None, None, d
        while True:
            kids = list(cur.items()) if isinstance(cur, dict) else list(enumerate(cur)) if isinstance(cur, list) else []
            if not kids or (parent is not None and rnd.random() < 0.3): return parent, key
            key, nxt = rnd.choice(kids)
            parent, cur = cur, nxt

    bad_ = 0
    with tempfile.TemporaryDirectory() as wf:
        open(os.path.join(wf, "w.yml"), "w").write("")
        real_load, real_mode, real_ignored = MOD.load, MOD.index_mode, MOD.ignored
        root = repo_for(_jobs[0][2], *_jobs[0][3])
        try:
            import functools
            MOD.index_mode, MOD.ignored = functools.lru_cache(None)(MOD.index_mode), functools.lru_cache(None)(MOD.ignored)  # (the same few paths: git is asked once)
            for k in range(n):
                doc = base()
                for _ in range(rnd.randint(1, 4)):
                    tgt, key = pick(doc)
                    if tgt is None: continue
                    if rnd.random() < 0.15 and isinstance(tgt, dict): del tgt[key]
                    else: tgt[key] = copy.deepcopy(rnd.choice(junk))
                MOD.load = lambda f, doc=doc: copy.deepcopy(doc)  # (parsing is tested by the cases above; here it is the checks that get odd values)
                r = run_check_in(root, wf)
                if "Traceback" in r.stderr:
                    bad_ += 1
                    if bad_ <= 3: print("FAIL fuzz\n" + yaml.safe_dump(doc) + r.stderr)
        finally:  # (the module is shared with the cases above and below)
            MOD.load, MOD.index_mode, MOD.ignored = real_load, real_mode, real_ignored
    print(("ok   " if not bad_ else "FAIL ") + f"structural fuzz: {n} documents, {bad_} tracebacks")
    return 1 if bad_ else 0


bad += fuzz(10000)
import shutil
for d_ in _repos.values(): shutil.rmtree(d_, ignore_errors=True)
print("test_check_workflows: " + ("ok" if not bad else f"{bad} failed"))
sys.exit(1 if bad else 0)
