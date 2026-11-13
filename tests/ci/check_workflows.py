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
    skipped. Also followed: `$(...)`/backticks, `eval`, `find -exec`, `pushd`/`popd`, `function f {..}`, `coproc`, `env -C/--chdir/-S`,
    `xargs -I{} sh -c '{}' <<< path` (the replacement string of a shell `-c` script is each input line), and heredocs /
    here-strings that a shell reads as its script (`bash <<EOF`, `bash - <<<`, `sh -s`; not `bash script.sh <<<`, which is only
    the script's stdin), also through pipes that pass the text on unchanged (`cat <<EOF | tee f | bash`; not through grep, sed,
    python3, `cat -n`, `cat file`, or a command whose stdin or stdout is redirected). Also: literal `NAME=value` / `export` /
    `env:` variables, and the prefixes env sudo xargs timeout nice ionice taskset time command exec nohup stdbuf setsid ccache
    valgrind strace ltrace (with their options, also by full path). Principle: when in doubt, no finding (a false positive breaks
    the CI). Not seen (false negatives): variables whose value is only known at run time (`${{ }}`, step outputs, files),
    `source`d scripts, `trap`, aliases, `((..))` arithmetic after other words than for/while/if/elif/until/then/do/else/!/time, `bash <<EOF >log` (output redirected: not followed), scripts nested absurdly deep (a WARN line, skipped), and whatever a
    script that is itself started with `sh`/`python3` runs. Git failing to answer is itself a failure. A workflow of an
    unexpected shape (not a mapping, `steps:` not a list, `run:` not text, a file that cannot be read or is not UTF-8) is reported
    in a line, never as a traceback.
"""
import datetime, glob, os, re, subprocess, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, ".github", "workflows")
bad = 0


def fail(f, msg):
    global bad
    bad += 1
    print(f"FAIL {os.path.basename(f)}: {msg}")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    except (OSError, ValueError):  # (no git; a NUL byte in a path)
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
        if c == "(" and script.startswith("((", i) and not have and (not toks or toks[-1][0] == "op" and toks[-1][1] != ")" or toks[-1] in [("w", x) for x in ("for", "while", "if", "elif", "until", "then", "do", "else", "!", "time")]):
            i = balanced(script, i, "(", ")"); continue  # an arithmetic command / loop header: `<<` in it is a shift, not a heredoc
        if c in ";|&()":
            end_word()
            if script.startswith(";;&", i): toks.append(("op", ";;")); i += 3  # case: the next pattern follows
            elif script.startswith(";&", i): toks.append(("op", ";;")); i += 2
            elif script[i:i + 2] in ("&&", "||", ";;"): toks.append(("op", script[i:i + 2])); i += 2
            elif script.startswith("|&", i): toks.append(("op", "|")); i += 2  # (also pipes stderr)
            else: toks.append(("op", c)); i += 1
            continue
        if c in "<>":
            fd = "".join(word) if have and "".join(word).isdigit() else None  # `2>log`: the digits are the descriptor, not a word
            if fd is not None: word, have = [], False
            end_word()
            if script.startswith(("<(", ">("), i):  # process substitution: its command runs, and the word stands for a file
                j = balanced(script, i + 1, "(", ")")
                toks.append(("sub", script[i + 2:max(i + 2, j - 1)])); toks.append(("w", "/dev/fd/63")); i = j; continue
            if script.startswith("<<<", i):  # here-string: the next word is the text fed to the command
                i += 3; toks.append(("hs",)); continue
            if script.startswith("<<", i):
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
            dup = i < n and script[i] == "&"  # `>&2`, `2>&1`: a descriptor copied, not a file
            while i < n and script[i] in "<>&|": i += 1
            target = re.match(r"\d+|-", script[i:]) if dup else None
            # what the redirection does to the command's stdin / stdout (a heredoc or a pipe no longer reaches it / goes on):
            # stdout is changed by `>f`, `>>f`, `>&2`, `1>&2`, `>&-`, not by `2>f`, `2>&1` or `>&1`
            kind = "in" if c == "<" else ("out" if fd in (None, "1") and not (target and target.group() == "1") else "other")
            toks.append(("redir", kind)); continue
        word.append(c); have = True; i += 1
    end_word()
    return toks, bodies


PREFIX_SKIP = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}", "builtin", "in", "esac"}
# Commands that run another command: (options that take a separate value, positional arguments before the command).
WRAP = {"env": ({"-u", "-C", "-S"}, 0), "xargs": (set(), 0),  # (env and xargs have their own option parsing)
        "sudo": ({"-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-U", "-r", "-t", "--user", "--group", "--host", "--prompt", "--chdir", "--role", "--type"}, 0),
        "timeout": ({"-s", "-k", "--signal", "--kill-after"}, 1), "nice": ({"-n", "--adjustment"}, 0),
        "ionice": ({"-c", "-n", "-p"}, 0), "taskset": (set(), 1), "time": (set(), 0), "command": (set(), 0), "exec": ({"-a"}, 0),
        "nohup": (set(), 0), "stdbuf": (set(), 0), "setsid": (set(), 0), "ccache": (set(), 0), "valgrind": (set(), 0),
        "strace": ({"-o", "-e", "-p", "-s", "-u", "-E", "-O", "-a", "-b", "-P", "-X", "-U"}, 0),
        "ltrace": ({"-o", "-e", "-p", "-s", "-u", "-a", "-n"}, 0)}
PASSTHROUGH = {"cat", "tee", "pv", "sponge"}
XARGS_VALUE = set("InPLsdEa")  # xargs short options that take a value (attached or the next word); -i and -e take an optional attached one
XARGS_LONG = {"--arg-file", "--max-args", "--max-procs", "--max-lines", "--max-chars", "--delimiter"}  # (--replace and --eof only attached)
SHELLS = {"bash", "sh", "dash", "zsh"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WORKSPACE = re.compile(r"^(\$\{\{\s*github\.workspace\s*\}\}|\$GITHUB_WORKSPACE|\$\{GITHUB_WORKSPACE\})(/|$)")
VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def shell_options(args):
    """(-c given, -s given, the first operand or None) of the arguments of bash/sh: options are the words before the first operand
    (a word that does not start with - or +; what follows is for the script), up to `--`; -o / -O / --rcfile / --init-file take a value."""
    has_c = has_s = False
    k = 0
    while k < len(args):
        a = args[k]
        if a == "--": return has_c, has_s, args[k + 1] if k + 1 < len(args) else None
        if a == "-" or not a or a[0] not in "-+": return has_c, has_s, a
        if a[:2] != "--" and a[0] == "-":
            has_c = has_c or "c" in a[1:]
            has_s = has_s or "s" in a[1:]
        k += 2 if a in ("--rcfile", "--init-file") or (a[:2] != "--" and a[-1] in "oO" and len(a) >= 2) else 1  # (a trailing o / O: the option name or shopt name follows)
    return has_c, has_s, None


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


def commands(script, cwd, env=None, stdin=None):
    """Yields (command word, cwd) for each command of the script that may be a path; cwd None = unknown."""
    env = dict(env or {})
    stack, dirs, cur, skip_target = [], [], [], False
    toks, bodies = lex(script)
    toks = case_filter(toks)
    out = []
    cur_hd, carry = None, stdin  # heredoc / here-string text of the current command / piped into the next one (stdin: what the script itself is fed)
    hs_next = False
    redir_break = False

    def flush(op=None):
        nonlocal cur, cwd, cur_hd, carry, redir_break
        if not cur and cur_hd is None and not redir_break:
            return  # nothing here (a newline after `|`: the pipe goes on to the next line)
        words, cur = cur, []
        hd, cur_hd = cur_hd, None  # (a heredoc or here-string)
        broken, redir_break = redir_break, False  # an `<` or `>` redirection of this command
        fed, carry = carry, None  # what an earlier command of the pipeline piped in (a heredoc)
        i = 0
        cmd_cwd, noexec, split_string, last_wrap, repl, wraps, from_file = cwd, False, None, None, None, [], False
        while i < len(words):
            w = words[i]
            if w in PREFIX_SKIP or ASSIGN.match(w): i += 1; continue
            if w == "function": i += 2; continue
            if w == "coproc":
                i += 1
                if i + 1 < len(words) and words[i + 1] == "{" and re.fullmatch(r"[A-Za-z_]\w*", words[i]): i += 1
                continue
            if os.path.isabs(w) and os.path.basename(w) in WRAP: w = os.path.basename(w)  # /usr/bin/env bash
            if w in WRAP:
                optargs, npos = WRAP[w]
                last_wrap = w
                wraps.append(w)
                i += 1
                while i < len(words):
                    a = words[i]
                    if ASSIGN.match(a): i += 1; continue
                    if a.startswith("-") and a != "-":
                        i += 1
                        if w == "command" and re.fullmatch(r"-[A-Za-z]*[vV][A-Za-z]*", a): noexec = True  # only looks the name up (-v, -pv ...)
                        val = None
                        if w == "xargs":  # -I STR / -ISTR / -i[STR] / --replace[=STR]: STR in the arguments stands for the input; -a FILE: the input is FILE
                            if a.startswith("--"):
                                opt, eq, att = a.partition("=")
                                if opt == "--arg-file": from_file = True
                                if opt == "--replace": repl = att or "{}"
                                if opt in XARGS_LONG and not eq and i < len(words): i += 1
                            else:
                                for k, ch in enumerate(a[1:], 1):  # (grouped: -ri, -ria FILE, -aFILE): the first option with a value ends the group
                                    att = a[k + 1:]
                                    if ch in "ie":
                                        if ch == "i": repl = att or "{}"
                                        break
                                    if ch in XARGS_VALUE:
                                        if ch == "a": from_file = True
                                        val = att if att else (words[i] if i < len(words) else None)
                                        if not att and i < len(words): i += 1
                                        if ch == "I": repl = val
                                        break
                            continue
                        if w == "env":
                            if a.startswith("--"):
                                opt, eq, att = a.partition("=")
                                att = att if eq else None
                            else:  # short options may be grouped (-iC DIR, -iCDIR, -u VAR, -S'...'): the first one that takes a value ends the group
                                opt, att = a, None
                                for k, ch in enumerate(a[1:], 1):
                                    if ch in "CuS": opt, att = "-" + ch, (a[k + 1:] or None); break
                            if opt in ("-C", "--chdir", "-u", "--unset", "-S", "--split-string"):
                                val = att
                                if val is None and i < len(words): val = words[i]; i += 1  # the value is the next word
                                if opt in ("-C", "--chdir") and val is not None:  # env -C DIR cmd: cmd runs in DIR
                                    cmd_cwd = None if cmd_cwd is None or "$" in val else os.path.normpath(os.path.join(cmd_cwd, val))
                                elif opt in ("-S", "--split-string"):
                                    split_string = val  # the string is split into the command and its arguments
                        elif a in optargs and i < len(words): i += 1
                        continue
                    break
                i += npos
                continue
            break
        if noexec:
            return
        if split_string is not None:
            out.extend(commands(split_string + " " + " ".join(words[i:]), cmd_cwd, env, hd or fed))  # (its stdin is the command's)
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
            has_c, has_s, operand = shell_options(args)
            if has_c and operand is not None:
                text = hd or fed
                if "xargs" in wraps and repl and repl in operand and text and not from_file:
                    # `xargs -I{} sh -c '{}' <<< tests/a/bad.sh`: the replacement string of the script is each input line (xargs never replaces the command's name)
                    for line in text.splitlines():
                        if line.strip(): out.extend(commands(operand.replace(repl, line.strip()), cmd_cwd, env))
                else:
                    out.extend(commands(operand, cmd_cwd, env))
                return
            # The text is the script only where the shell reads its script from stdin: no operand (a file) or `-` / /dev/stdin, or `-s`.
            from_stdin = has_s or operand is None or operand in ("-", "/dev/stdin")
            if (hd or fed) and "xargs" not in wraps and from_stdin and not broken:  # (xargs feeds the shell arguments) `bash <<EOF` / `cat <<EOF | bash`
                out.extend(commands(hd or fed, cmd_cwd, env))
            return
        # The text goes on through a pipe only past commands that pass their input on unchanged (cat, tee ...): grep, sed, python3 ... transform it.
        # (cat with a file argument reads that file, not its stdin; a `<` or `>` redirection of the command takes the text away or off the pipe.)
        files = [a for a in args if not a.startswith("-") or a == "-"]
        if (hd or fed) and op == "|" and not broken and os.path.basename(w) in PASSTHROUGH and (os.path.basename(w) == "tee" or (all(a == "-" for a in files) and (os.path.basename(w) != "cat" or all(not a.startswith("-") or a == "-" for a in args)))):  # (cat -n, -A, -v ... change the text)
            carry = hd or fed
        out.append((expand(w, env), cmd_cwd))

    for t in toks:
        if t[0] == "sub":
            out.extend(commands(t[1], cwd, env))
        elif t[0] == "hd":
            cur_hd = bodies[t[1]]
        elif t[0] == "hs":
            hs_next = True
        elif t[0] == "w":
            if hs_next: cur_hd = t[1]; hs_next = False
            elif skip_target: skip_target = False
            else: cur.append(t[1])
        elif t[0] == "redir":
            skip_target = True
            if t[1] == "out" or (t[1] == "in" and cur_hd is not None): redir_break = True  # (`cat <x <<EOF`: the heredoc, being later, wins)
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
    try:
        found = commands(script, base_cwd, dict(env or {}, GITHUB_WORKSPACE=ROOT))
    except RecursionError:  # absurdly nested (`eval eval ...` thousands deep): not analysed
        print("WARN a script is nested too deeply to be analysed (skipped)")
        return problems
    for w, cwd in found:
        m = WORKSPACE.match(w)
        if m: w, cwd = w[m.end():] or ".", ROOT
        if "$" in w or "`" in w or "\0" in w or "/" not in w or cwd is None or any(ch in w for ch in "*?["):
            continue
        full = os.path.normpath(w if os.path.isabs(w) else os.path.join(cwd, w))
        rel = os.path.relpath(full, ROOT)
        if rel.startswith("..") or "\0" in rel:
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


def mapping(f, what, x):
    """x as a dict: None / empty -> {}; anything else that is not a mapping is reported and ignored."""
    if isinstance(x, dict): return x
    if x: fail(f, f"{what} is not a mapping (ignored)")
    return {}


def step_list(f, what, x):
    """The steps (dicts) of a `steps:` value; null / empty -> none; entries that are not mappings are reported and ignored."""
    if not x: return []
    if not isinstance(x, list):
        fail(f, f"{what}: steps is not a list (ignored)")
        return []
    out = []
    for k, st in enumerate(x):
        if isinstance(st, dict): out.append(st)
        elif st is not None: fail(f, f"{what}: step {k + 1} is not a mapping (ignored)")
    return out


def plain_env(f, *maps):
    """The `env:` maps (workflow, job, step) as a dict of literal strings (values with ${{ }} are unknown)."""
    out = {}
    for m in maps:
        for k, v in mapping(f, "env", m).items():
            if "${{" in str(v): out.pop(str(k), None)
            else: out[str(k)] = str(v)
    return out


def run_text(f, label, step):
    """The step's `run:` as text (a number is its digits); None for no run, or one that is not a scalar (reported)."""
    run = step.get("run")
    if run is None or run == "": return None
    if isinstance(run, bool): return str(run).lower()  # (YAML true: the command `true`)
    if isinstance(run, (str, int, float, datetime.date)): return str(run)  # (YAML reads 2024-01-01 as a date, 1e3 as a number: they are text to the shell)
    fail(f, f"{label}: step {step.get('name', '?')!r} has a run that is not text (ignored)")
    return None


def check_steps(f, label, steps, base, shell=None, env=None):
    for step in steps:
        run = run_text(f, label, step)
        if not run or not is_bash(step.get("shell") or shell or "bash"):
            continue  # (pwsh, python, cmd: not shell syntax)
        wd = step.get("working-directory", base)
        cwd = ROOT if not wd else (None if "$" in str(wd) else os.path.normpath(os.path.join(ROOT, str(wd))))
        for pr in check_direct(run, cwd, plain_env(f, env, step.get("env"))):
            fail(f, f"{label}: step {step.get('name', '?')!r} {pr}")


def run_default(d, key):
    """defaults.run.KEY of a workflow or job (anything that is not a mapping on the way is no default)."""
    dflt = d.get("defaults") if isinstance(d, dict) else None
    run = dflt.get("run") if isinstance(dflt, dict) else None
    return run.get(key) if isinstance(run, dict) else None


def load(f):
    try:
        with open(f, "rb") as fh: text = fh.read().decode("utf-8")
        return yaml.safe_load(text)
    except UnicodeDecodeError as e:
        fail(f, f"is not UTF-8 text: {e}")
    except OSError as e:  # (a directory called x.yml, no permission ...)
        fail(f, f"cannot be read: {e}")
    except yaml.YAMLError as e:
        fail(f, f"does not parse: {e}")
    except RecursionError:
        fail(f, "does not parse: nested too deeply")
    return None


def main(argv):
    """Checks the workflows of argv[1] (default .github/workflows) of the repo argv[2] (default: this one); returns the exit code."""
    global ROOT, DIR, bad
    if len(argv) > 2: ROOT = os.path.abspath(argv[2])
    DIR = argv[1] if len(argv) > 1 else os.path.join(ROOT, ".github", "workflows")
    bad = 0
    check_all()
    print("check_workflows: ok" if not bad else f"check_workflows: {bad} problems")
    return 1 if bad else 0


INJECT = re.compile(r"\$\{\{\s*(inputs\.|github\.event\.|github\.head_ref)")


def check_all():
    for f in sorted(glob.glob(os.path.join(DIR, "*.yml"))):
        wf = load(f)
        if wf is None: continue  # (an empty file, or one that did not parse: reported above)
        wf = mapping(f, "the workflow", wf)
        conc = wf.get("concurrency")
        if conc is not None:
            group = str(conc.get("group", "")) if isinstance(conc, dict) else str(conc)
            if "github.run_id" not in group and "head_commit.message" not in group:
                fail(f, f"workflow-level concurrency group {group!r} is shared with the runs whose jobs are all skipped")
        jobs = {n: mapping(f, f"job {n}", j) for n, j in mapping(f, "jobs", wf.get("jobs")).items()}
        steps_of = {n: step_list(f, f"job {n}", j.get("steps")) for n, j in jobs.items()}
        releases = any("action-gh-release" in str(s.get("uses", "")) for st in steps_of.values() for s in st)
        if releases and conc is None:
            fail(f, "publishes a release without a concurrency group")
        for name, job in jobs.items():
            jc = job.get("concurrency")
            if os.path.basename(f) == "gles-cts.yml" and isinstance(jc, dict) and jc.get("cancel-in-progress") is True:
                g = str(jc.get("group", ""))
                if "github.ref" not in g or "github.run_id" not in g:
                    fail(f, f"job {name}: concurrency group {g!r} must use github.ref (manual runs) and github.run_id (the rest)")
            windows = "windows" in str(job.get("runs-on", "")).lower()  # (the default shell there is pwsh)
            check_steps(f, f"job {name}", steps_of[name], run_default(job, "working-directory") or run_default(wf, "working-directory"),
                        run_default(job, "shell") or run_default(wf, "shell") or ("pwsh" if windows else None),
                        plain_env(f, wf.get("env"), job.get("env")))
            for step in steps_of[name]:
                run = step.get("run")
                if isinstance(run, str) and INJECT.search(run):
                    fail(f, f"job {name}: step {step.get('name', '?')!r} puts an input or event field inside the script")
    for f in sorted(glob.glob(os.path.join(ROOT, ".github", "actions", "*", "action.yml"))):
        act = load(f)
        if act is None: continue
        runs = mapping(f, "the action", act).get("runs")
        check_steps(f, "composite action", step_list(f, "composite action", mapping(f, "runs", runs).get("steps")), None)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
