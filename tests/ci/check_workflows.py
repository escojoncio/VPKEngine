#!/usr/bin/env python3
"""Static checks of .github/workflows and .github/actions/*/action.yml (PyYAML):
    python3 -I tests/ci/check_workflows.py [workflows dir [repo root]]

 1. every file parses;
 2. a workflow-level `concurrency` group must depend on `github.run_id` or on the commit message gate: a push without [build]
    skips every job, but its run would still take the group, cancel a running [build] run (cancel-in-progress) or replace
    the pending one (only one run can wait in a group);
 3. a workflow that publishes a release (softprops/action-gh-release) has a concurrency group, so two runs do not race for
    the same tag;
 4. no `${{ inputs.* }}` or `${{ github.event.* }}` text inside a `run:` script (shell injection through a dispatch input
    or a commit message): they go through `env:`.
 5. in gles-cts.yml a job with `cancel-in-progress: true` groups by ref (a newer manual run cancels the older one) and by
    run_id otherwise (a push, whose jobs are mostly skipped, cancels nothing and is cancelled by nothing): a group made of
    run_id alone cancels nothing;
 6. a command of a `run:` script that is a path of the repo (`./x.sh`, `cd dir && ./x.sh`, `working-directory:` + `./x.sh`,
    `$GITHUB_WORKSPACE/x`, behind `env`/`sudo`/`xargs`/`!`/`if`/`bash -c '...'`, after `;` `&&` `||` `|` `(`, heredoc bodies
    skipped) runs that file directly: if tracked it must have mode 100755 in the git index (a 100644 script ends in
    "Permission denied" on the runner); a path under tests/ tools/ gpu/ runtime/ that is neither tracked nor ignored by git
    (build outputs are) does not exist. `sh path` / `python3 path` are not affected. A symlink (mode 120000) is not judged.
    Steps whose `shell:` is not bash/sh are skipped. Not seen (false negatives): commands behind `eval`, `$(...)`, backticks,
    variables (`$X/run.sh`), functions or aliases, `find -exec`, `xargs -I{} X`, `timeout N X`, `sudo -u user X`, `here-strings`,
    and anything a script that is itself started with `sh`/`python3` runs. Git failing to answer is itself a failure.
"""
import glob, os, re, subprocess, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv) > 2:
    ROOT = os.path.abspath(sys.argv[2])
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".github", "workflows")
bad = 0


def fail(f, msg):
    global bad
    bad += 1
    print(f"FAIL {os.path.basename(f)}: {msg}")


_git_cache = {}


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    except OSError:
        return None


def index_mode(path):
    """The mode of `path` in the git index ('' if it is not tracked); None if there is no repository to ask."""
    r = _git("ls-files", "-s", "--", path)
    if r is None or r.returncode != 0:
        return None
    return r.stdout.split()[0] if r.stdout.split() else ""


def ignored(path):
    r = _git("check-ignore", "-q", "--", path)
    return r is not None and r.returncode == 0


# ---- a small shell lexer: enough to find the commands of a `run:` script ----
def lex(script):
    """Tokens ('w', word) / ('op', ';' '&&' '||' '|' '&' '(' ')') / ('redir',); a newline is ';'. Heredoc bodies are skipped."""
    toks, i, n = [], 0, len(script)
    word, have, pending = [], False, []

    def end_word():
        nonlocal word, have
        if have:
            toks.append(("w", "".join(word)))
        word, have = [], False

    def balanced(i, o, c):  # i at the opener; returns the index after the matching closer
        d = 0
        while i < n:
            if script[i] == o: d += 1
            elif script[i] == c:
                d -= 1
                if d == 0: return i + 1
            i += 1
        return n

    while i < n:
        c = script[i]
        if c == "\\":
            if i + 1 < n and script[i + 1] == "\n": i += 2; continue
            if i + 1 < n: word.append(script[i + 1]); have = True
            i += 2; continue
        if c == "'":
            j = script.find("'", i + 1); j = n if j < 0 else j
            word.append(script[i + 1:j]); have = True; i = j + 1; continue
        if c == '"':
            j = i + 1
            while j < n and script[j] != '"': j += 2 if script[j] == "\\" else 1
            word.append(script[i + 1:j]); have = True; i = j + 1; continue
        if c == "$" and script.startswith("${{", i):
            j = script.find("}}", i); j = n if j < 0 else j + 2
            word.append(script[i:j]); have = True; i = j; continue
        if c == "$" and script.startswith("$(", i):
            j = balanced(i + 1, "(", ")"); word.append(script[i:j]); have = True; i = j; continue
        if c == "$" and script.startswith("${", i):
            j = balanced(i + 1, "{", "}"); word.append(script[i:j]); have = True; i = j; continue
        if c == "`":
            j = script.find("`", i + 1); j = n if j < 0 else j + 1
            word.append(script[i:j]); have = True; i = j; continue
        if c == "#" and not have:
            j = script.find("\n", i); i = n if j < 0 else j; continue
        if c in " \t":
            end_word(); i += 1; continue
        if c == "\n":
            end_word(); toks.append(("op", ";")); i += 1
            for delim, strip in pending:  # heredoc bodies start on the next line
                while i < n:
                    j = script.find("\n", i); j = n if j < 0 else j
                    line = script[i:j]; i = min(n, j + 1)
                    if (line.lstrip("\t") if strip else line) == delim: break
            pending = []
            continue
        if c == "(" and have and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\+?=", "".join(word)):
            j = balanced(i, "(", ")")  # an array assignment, `X=(a b)` / `X+=(...)`, also over several lines: one word
            word.append(script[i:j]); i = j; continue
        if c in ";|&()":
            end_word()
            two = script[i:i + 2]
            if two in ("&&", "||", ";;"): toks.append(("op", two)); i += 2
            else: toks.append(("op", c)); i += 1
            continue
        if c in "<>":
            end_word()
            if script.startswith("<<", i) and not script.startswith("<<<", i):
                i += 2; strip = script.startswith("-", i); i += strip
                while i < n and script[i] in " \t": i += 1
                d = []
                while i < n and script[i] not in " \t\n;|&()<>":
                    if script[i] == "\\": i += 1
                    elif script[i] in "'\"": 
                        q = script[i]; j = script.find(q, i + 1); j = n if j < 0 else j
                        d.append(script[i + 1:j]); i = j + 1
                    else: d.append(script[i]); i += 1
                pending.append(("".join(d), strip)); toks.append(("redir",)); continue
            i += 1
            while i < n and script[i] in "<>&|": i += 1
            toks.append(("redir",)); continue
        word.append(c); have = True; i += 1
    end_word()
    return toks


PREFIX_SKIP = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}", "time", "exec", "command",
               "nohup", "builtin", "in", "esac"}
WRAPPERS = {"env", "sudo", "xargs", "nice", "stdbuf"}
SHELLS = {"bash", "sh", "dash", "zsh"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WORKSPACE = re.compile(r"^(\$\{\{\s*github\.workspace\s*\}\}|\$GITHUB_WORKSPACE|\$\{GITHUB_WORKSPACE\})(/|$)")


def case_filter(toks):
    """Drops the patterns of `case WORD in PATTERN) ... ;; ... esac` (globs, not commands)."""
    out, stack, words = [], [], []  # stack: "pat" (reading patterns) / "body" per open case; words: of the current command
    for t in toks:
        if stack and stack[-1] == "pat":
            if t == ("w", "esac") and not words:
                stack.pop(); out.append(t)
            elif t == ("op", ")"):
                stack[-1] = "body"; words = []; out.append(("op", ";"))  # the body is a new command
            elif t[0] == "op" and t[1] == ";":
                pass  # a newline between patterns
            continue
        if t[0] == "w":
            words.append(t[1])
            if t[1] == "in" and len(words) >= 3 and words[0] == "case":
                stack.append("pat"); words = []
            elif t[1] == "esac" and len(words) == 1 and stack:
                stack.pop()
        else:
            words = []
            if t == ("op", ";;") and stack:
                stack[-1] = "pat"
        out.append(t)
    return out


def commands(script, cwd):
    """Yields (command word, cwd) for each command of the script that may be a path; cwd None = unknown."""
    stack, cur, skip_target = [], [], False
    toks = case_filter(lex(script))
    out = []

    def flush():
        nonlocal cur, cwd
        words, cur = cur, []
        i = 0
        while i < len(words):
            w = words[i]
            if w in PREFIX_SKIP or ASSIGN.match(w): i += 1; continue
            if w in WRAPPERS:
                i += 1
                while i < len(words) and (words[i].startswith("-") or ASSIGN.match(words[i])): i += 1
                continue
            break
        if i >= len(words) or words[i] in ("for", "case", "select"):
            return
        w, args = words[i], words[i + 1:]
        if w == "cd":
            tgt = args[0] if args else ""
            if cwd is None or not tgt or "$" in tgt or tgt in ("-", "~"): cwd = None
            else: cwd = os.path.normpath(os.path.join(cwd, tgt))
            return
        if w in SHELLS or (os.path.isabs(w) and os.path.basename(w) in SHELLS):
            for k, a in enumerate(args):
                if re.match(r"^-[a-z]*c[a-z]*$", a) and k + 1 < len(args):
                    out.extend(commands(args[k + 1], cwd))
                    return
            return
        out.append((w, cwd))

    for t in toks:
        if t[0] == "w":
            if skip_target: skip_target = False
            else: cur.append(t[1])
        elif t[0] == "redir":
            skip_target = True
        else:
            skip_target = False
            op = t[1]
            if op == "(": flush(); stack.append(cwd)
            elif op == ")":
                flush()
                if stack: cwd = stack.pop()
            else: flush()
    flush()
    return out


def check_direct(script, base_cwd):
    """Problems (strings) of the commands of `script` that run a path of the repo directly."""
    problems = []
    for w, cwd in commands(script, base_cwd):
        m = WORKSPACE.match(w)
        if m: w, cwd = w[m.end():] or ".", ROOT
        if "$" in w or "`" in w or "/" not in w or cwd is None or any(ch in w for ch in "*?["):
            continue
        full = os.path.normpath(w if os.path.isabs(w) else os.path.join(cwd, w))
        rel = os.path.relpath(full, ROOT)
        if rel.startswith(".."):
            continue
        mode = index_mode(rel)
        if mode is None:
            problems.append(f"runs {rel} directly but git did not answer (is {ROOT} a git repository with git installed?)")
            continue
        if mode:
            if mode == "120000":
                continue  # a symlink: its target is not judged here
            if mode != "100755": problems.append(f"runs {rel} directly but its index mode is {mode} (needs 100755)")
        elif not ignored(rel) and not os.path.exists(full) and rel.split(os.sep)[0] in ("tests", "tools", "gpu", "runtime"):
            problems.append(f"runs {rel} directly but it is neither tracked nor ignored and does not exist")
    return problems


def check_steps(f, label, steps, base, shell=None):
    for step in steps:
        run = step.get("run")
        sh = str(step.get("shell") or shell or "bash")
        if not run or not (sh.startswith("bash") or sh.startswith("sh")):
            continue  # (pwsh, python, cmd: not shell syntax)
        wd = step.get("working-directory", base)
        cwd = ROOT if not wd else (None if "$" in str(wd) else os.path.normpath(os.path.join(ROOT, str(wd))))
        for pr in check_direct(run, cwd):
            fail(f, f"{label}: step {step.get('name', '?')!r} {pr}")


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
        jc = job.get("concurrency")
        if os.path.basename(f) == "gles-cts.yml" and isinstance(jc, dict) and jc.get("cancel-in-progress") is True:
            g = str(jc.get("group", ""))
            if "github.ref" not in g or "github.run_id" not in g:
                fail(f, f"job {name}: concurrency group {g!r} must use github.ref (manual runs) and github.run_id (the rest)")
        dflt = lambda d: ((d or {}).get("defaults") or {}).get("run", {}).get("working-directory")
        dsh = lambda d: ((d or {}).get("defaults") or {}).get("run", {}).get("shell")
        check_steps(f, f"job {name}", job.get("steps", []), dflt(job) or dflt(wf), dsh(job) or dsh(wf))
        for step in job.get("steps", []):
            run = step.get("run")
            if run and INJECT.search(run):
                fail(f, f"job {name}: step {step.get('name', '?')!r} puts an input or event field inside the script")
for f in sorted(glob.glob(os.path.join(ROOT, ".github", "actions", "*", "action.yml"))):
    try:
        act = yaml.safe_load(open(f))
    except yaml.YAMLError as e:
        fail(f, f"does not parse: {e}")
        continue
    check_steps(f, "composite action", ((act or {}).get("runs") or {}).get("steps", []), None)
print("check_workflows: ok" if not bad else f"check_workflows: {bad} problems")
sys.exit(1 if bad else 0)
