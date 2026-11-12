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
    Steps whose `shell:` is not bash/sh (and, without `shell:`, the steps of `runs-on: windows-*` jobs, which use pwsh) are
    skipped. Also followed: `$(...)`/backticks, `eval`, `find -exec`, `pushd`/`popd`, `function f {..}`, `coproc`, literal
    `NAME=value` / `export` / `env:` variables, and the prefixes env sudo xargs timeout nice ionice taskset time command exec nohup
    stdbuf setsid ccache valgrind strace ltrace (with their options). Not seen (false negatives): variables whose value is only
    known at run time (`${{ }}`, step outputs, files), `source`d scripts, `trap`, aliases, here-strings, and whatever a script
    that is itself started with `sh`/`python3` runs. Git failing to answer is itself a failure.
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
def ansi_end(s, i):
    """s[i] is the quote of a $'...' string (backslash escapes); the index after the closing quote."""
    j, n = i + 1, len(s)
    while j < n and s[j] != "'": j += 2 if s[j] == "\\" else 1
    return min(n, j + 1)


def dq_scan(s, i):
    """s[i] is a double quote; (the index after the closing one, whether it was found); $(...) and ${...} inside are units."""
    j, n = i + 1, len(s)
    while j < n and s[j] != '"':
        if s[j] == "\\": j += 2
        elif s.startswith("$(", j): j = balanced(s, j + 1, "(", ")")
        elif s.startswith("${", j): j = balanced(s, j + 1, "{", "}", True)
        else: j += 1
    return (j + 1, True) if j < n else (n, False)


def dq_end(s, i):
    return dq_scan(s, i)[0]


def scan(s, i, o, c, dq=False):
    """s[i] is the opener o: (index after the matching closer c, whether it was found). Quotes ('..', "..", $'..') and
    backslashes are respected; dq: the text is inside double quotes (a single quote is a plain character there); inside $( ) a
    `#` that starts a word is a comment (its quotes do not count)."""
    n, d = len(s), 0
    while i < n:
        ch = s[i]
        if ch == "\\": i += 2; continue
        if ch == "'" and not dq:
            j = s.find("'", i + 1); i = n if j < 0 else j + 1; continue
        if ch == '"': i = dq_end(s, i); continue
        if ch == "$" and not dq and s.startswith("$'", i): i = ansi_end(s, i + 1); continue
        if ch == "#" and o == "(" and (i == 0 or s[i - 1] in " \t\n;&|("):
            j = s.find("\n", i); i = n if j < 0 else j; continue
        if ch == o: d += 1
        elif ch == c:
            d -= 1
            if d == 0: return i + 1, True
        i += 1
    return n, False


def balanced(s, i, o, c, dq=False):
    return scan(s, i, o, c, dq)[0]


def inner(s, i, j, closed):
    """The text of s[i:j] without the 2-character opener `$(` / `${` and, if the construct was closed, its closer."""
    return s[i + 2:j - 1] if closed else s[i + 2:j]


def subs_in(text):
    """The scripts of the $(...) and `...` substitutions of text (text is not single-quoted)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == "\\": i += 2; continue
        if text.startswith("$(", i) and not text.startswith("$((", i):
            j, closed = scan(text, i + 1, "(", ")"); out.append(inner(text, i, j, closed)); i = j; continue
        if text[i] == "`":
            j = text.find("`", i + 1); j = n if j < 0 else j; out.append(text[i + 1:j]); i = j + 1; continue
        i += 1
    return out


def lex(script):
    """Tokens ('w', word) / ('op', ';' '&&' '||' '|' '&' '(' ')' ';;') / ('redir',) / ('sub', script of a substitution) / ('hd', index of
    a heredoc body); a newline is ';'. Returns (tokens, heredoc bodies)."""
    toks, i, n = [], 0, len(script)
    word, have, pending, bodies = [], False, [], []

    def end_word():
        nonlocal word, have
        if have:
            toks.append(("w", "".join(word)))
        word, have = [], False

    while i < n:
        c = script[i]
        if c == "\\":
            if i + 1 < n and script[i + 1] == "\n": i += 2; continue
            if i + 1 < n: word.append(script[i + 1]); have = True
            i += 2; continue
        if c == "'":
            j = script.find("'", i + 1); j = n if j < 0 else j
            word.append(script[i + 1:j]); have = True; i = j + 1; continue
        if c == "$" and script.startswith("$'", i):
            j = ansi_end(script, i + 1); word.append(script[i + 2:max(i + 2, j - 1)]); have = True; i = j; continue
        if c == '"':
            j, closed = dq_scan(script, i); text = script[i + 1:j - 1] if closed else script[i + 1:j]
            word.append(text); have = True; i = j
            toks.extend(("sub", x) for x in subs_in(text))
            continue
        if c == "$" and script.startswith("${{", i):
            j = script.find("}}", i); j = n if j < 0 else j + 2
            word.append(script[i:j]); have = True; i = j; continue
        if c == "$" and script.startswith("$(", i):
            j = scan(script, i + 1, "(", ")")[0]; word.append(script[i:j]); have = True
            toks.extend(("sub", x) for x in subs_in(script[i:j])); i = j; continue
        if c == "$" and script.startswith("${", i):
            j = balanced(script, i + 1, "{", "}"); word.append(script[i:j]); have = True; i = j; continue
        if c == "`":
            j = script.find("`", i + 1); j = n if j < 0 else j + 1
            word.append(script[i:j]); have = True; toks.extend(("sub", x) for x in subs_in(script[i:j])); i = j; continue
        if c == "#" and not have:
            j = script.find("\n", i); i = n if j < 0 else j; continue
        if c in " \t":
            end_word(); i += 1; continue
        if c == "\n":
            end_word(); toks.append(("op", ";")); i += 1
            for delim, strip, idx in pending:  # heredoc bodies start on the next line
                body = []
                while i < n:
                    j = script.find("\n", i); j = n if j < 0 else j
                    line = script[i:j]; i = min(n, j + 1)
                    if (line.lstrip("\t") if strip else line) == delim: break
                    body.append(line)
                bodies[idx] = "\n".join(body)
            pending = []
            continue
        if c == "(" and have and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\+?=", "".join(word)):
            j = balanced(script, i, "(", ")")  # an array assignment, `X=(a b)` / `X+=(...)`, also over several lines: one word
            word.append(script[i:j]); i = j; continue
        if c in ";|&()":
            end_word()
            if script.startswith(";;&", i): toks.append(("op", ";;")); i += 3  # case: the next pattern follows
            elif script.startswith(";&", i): toks.append(("op", ";;")); i += 2
            elif script[i:i + 2] in ("&&", "||", ";;"): toks.append(("op", script[i:i + 2])); i += 2
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
                pending.append(("".join(d), strip, len(bodies))); toks.append(("hd", len(bodies))); bodies.append(""); continue
            i += 1
            while i < n and script[i] in "<>&|": i += 1
            toks.append(("redir",)); continue
        word.append(c); have = True; i += 1
    end_word()
    return toks, bodies


PREFIX_SKIP = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}", "builtin", "in", "esac"}
# Commands that run another command: (options that take a separate value, positional arguments before the command).
WRAP = {"env": ({"-u", "-C", "-S"}, 0), "sudo": ({"-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-U", "-r", "-t"}, 0),
        "xargs": ({"-I", "-n", "-P", "-L", "-s", "-d", "-E", "-a"}, 0), "timeout": ({"-s", "-k"}, 1), "nice": ({"-n"}, 0),
        "ionice": ({"-c", "-n", "-p"}, 0), "taskset": (set(), 1), "time": (set(), 0), "command": (set(), 0), "exec": ({"-a"}, 0),
        "nohup": (set(), 0), "stdbuf": (set(), 0), "setsid": (set(), 0), "ccache": (set(), 0), "valgrind": (set(), 0),
        "strace": ({"-o", "-e", "-p", "-s", "-u", "-E", "-O", "-a", "-b", "-P", "-X", "-U"}, 0),
        "ltrace": ({"-o", "-e", "-p", "-s", "-u", "-a", "-n"}, 0)}
SHELLS = {"bash", "sh", "dash", "zsh"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WORKSPACE = re.compile(r"^(\$\{\{\s*github\.workspace\s*\}\}|\$GITHUB_WORKSPACE|\$\{GITHUB_WORKSPACE\})(/|$)")
VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def expand(w, env):
    """w with the known $NAME / ${NAME} replaced (unknown ones stay)."""
    return VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), w)


def case_filter(toks):
    """Drops the patterns of `case WORD in PATTERN) ... ;; ... esac` (globs, not commands)."""
    out, stack, words = [], [], []  # stack: "pat" (reading patterns) / "body" per open case; words: of the current command
    for t in toks:
        if t[0] in ("sub", "hd"):
            out.append(t); continue
        if stack and stack[-1] == "pat":
            if t == ("w", "esac") and not words:
                stack.pop(); out.append(t)
            elif t == ("op", ")"):
                stack[-1] = "body"; words = []; out.append(("op", ";"))  # the body is a new command
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


def commands(script, cwd, env=None):
    """Yields (command word, cwd) for each command of the script that may be a path; cwd None = unknown."""
    env = dict(env or {})
    stack, dirs, cur, skip_target = [], [], [], False
    toks, bodies = lex(script)
    toks = case_filter(toks)
    out = []
    cur_hd = carry = None  # heredoc body of the current command / piped into the next one

    def flush(op=None):
        nonlocal cur, cwd, cur_hd, carry
        words, cur = cur, []
        hd, cur_hd = cur_hd, None
        fed, carry = carry, None  # what an earlier command of the pipeline piped in (a heredoc)
        i = 0
        cmd_cwd, noexec = cwd, False
        while i < len(words):
            w = words[i]
            if w in PREFIX_SKIP or ASSIGN.match(w): i += 1; continue
            if w == "function": i += 2; continue
            if w == "coproc":
                i += 1
                if i + 1 < len(words) and words[i + 1] == "{" and re.fullmatch(r"[A-Za-z_]\w*", words[i]): i += 1
                continue
            if w in WRAP:
                optargs, npos = WRAP[w]
                i += 1
                while i < len(words):
                    a = words[i]
                    if ASSIGN.match(a): i += 1; continue
                    if a.startswith("-") and a != "-":
                        i += 1
                        if w == "command" and a in ("-v", "-V"): noexec = True  # only looks the name up
                        if w == "env" and a.startswith("--chdir="): a, val = "-C", a[8:]
                        elif a in optargs and i < len(words): val = words[i]; i += 1
                        else: val = None
                        if w == "env" and a == "-C" and val is not None:  # env -C DIR cmd: cmd runs in DIR
                            cmd_cwd = None if cmd_cwd is None or "$" in val else os.path.normpath(os.path.join(cmd_cwd, val))
                        continue
                    break
                i += npos
                continue
            break
        if noexec:
            return
        if i >= len(words):  # only assignments: remember them for the $VARs of later commands
            for a in words:
                m = ASSIGN.match(a)
                if m and not a.split("=", 1)[0].endswith("+"): env[a.split("=", 1)[0]] = expand(a.split("=", 1)[1], env)
            return
        if words[i] in ("for", "case", "select"):
            return
        if words[i] == "export":
            for a in words[i + 1:]:
                if ASSIGN.match(a): env[a.split("=", 1)[0]] = expand(a.split("=", 1)[1], env)
            return
        w, args = words[i], words[i + 1:]
        if w in ("cd", "pushd", "popd"):
            if w == "popd":
                cwd = dirs.pop() if dirs else None
                return
            tgt = args[0] if args else ""
            if w == "pushd": dirs.append(cwd)
            if cwd is None or not tgt or "$" in tgt or tgt in ("-", "~") or tgt[0] in "+-": cwd = None
            else: cwd = os.path.normpath(os.path.join(cwd, tgt))
            return
        if w == "eval":
            out.extend(commands(" ".join(args), cmd_cwd, env))
            return
        if w == "find":
            for k, a in enumerate(args):
                if a in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(args): out.append((expand(args[k + 1], env), cmd_cwd))
            return
        if w in SHELLS or (os.path.isabs(w) and os.path.basename(w) in SHELLS):
            for k, a in enumerate(args):
                if re.match(r"^-[a-z]*c[a-z]*$", a) and k + 1 < len(args):
                    out.extend(commands(args[k + 1], cmd_cwd, env))
                    return
            if hd or fed:  # `bash <<EOF` / `cat <<EOF | bash`: the heredoc is the script it runs
                out.extend(commands(hd or fed, cmd_cwd, env))
            return
        if hd and op == "|": carry = hd
        out.append((expand(w, env), cmd_cwd))

    for t in toks:
        if t[0] == "sub":
            out.extend(commands(t[1], cwd, env))
        elif t[0] == "hd":
            cur_hd = bodies[t[1]]
        elif t[0] == "w":
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
            else: flush(op)
    flush()
    return out


def check_direct(script, base_cwd, env=None):
    """Problems (strings) of the commands of `script` that run a path of the repo directly."""
    problems = []
    for w, cwd in commands(script, base_cwd, dict(env or {}, GITHUB_WORKSPACE=ROOT)):
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


def is_bash(shell):
    words = str(shell).split()
    return bool(words) and os.path.basename(words[0]) in SHELLS


def plain_env(*maps):
    """The `env:` maps (workflow, job, step) as a dict of literal strings (values with ${{ }} are unknown)."""
    out = {}
    for m in maps:
        for k, v in (m or {}).items():
            if "${{" in str(v): out.pop(str(k), None)
            else: out[str(k)] = str(v)
    return out


def check_steps(f, label, steps, base, shell=None, env=None):
    for step in steps:
        run = step.get("run")
        if not run or not is_bash(step.get("shell") or shell or "bash"):
            continue  # (pwsh, python, cmd: not shell syntax)
        wd = step.get("working-directory", base)
        cwd = ROOT if not wd else (None if "$" in str(wd) else os.path.normpath(os.path.join(ROOT, str(wd))))
        for pr in check_direct(run, cwd, plain_env(env, step.get("env"))):
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
        windows = "windows" in str(job.get("runs-on", "")).lower()  # (the default shell there is pwsh)
        check_steps(f, f"job {name}", job.get("steps", []), dflt(job) or dflt(wf), dsh(job) or dsh(wf) or ("pwsh" if windows else None),
                    plain_env(wf.get("env"), job.get("env")))
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
