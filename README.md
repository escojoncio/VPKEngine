# VPEngine

A native engine for running PlayStation 4 games on Apple Vision Pro, built from the pieces that
the AstroVisionPro and PTVisionPro ports had to invent one at a time:

- **`tools/vpaot`** — an ahead-of-time translator: the game's x86-64 executable becomes C, which
  clang compiles into the app as ordinary arm64 code. No JIT at run time, so no debugger
  attachment, no executable memory, and the translated code is clean, signed pages that do not
  count against the process's memory limit. See [docs/PLAN.md](docs/PLAN.md).
- **`runtime/`** — the x86-64 machine state and instruction semantics the translated C calls
  (header-only, exact flags, SSE), and a small host that enters translated code.
- **`platform/visionos/`** — the Swift layer every game app shares: the VPS4 folder, controllers
  (DualSense, DualShock 4, PlayStation VR2 Sense with tracking), logs, language.
- **`tests/aot/`** — differential tests: every case runs natively on an x86-64 host and as
  translated C, from the same state, and the results must match bit for bit. The arm64 CI job
  then runs the translated C on ARM against the states the x86 job recorded.

Status: the translator handles the integer, control-flow and SSE instructions compilers emit
(99.6 % of the instructions of a real `/bin/ls`), jump tables, direct and indirect calls, and
passes the differential suite. What a whole game still needs is listed in
[docs/PLAN.md](docs/PLAN.md). No game data is included or needed to build or test.

## Build and test

```sh
git clone --recursive <this repository> && cd VPEngine
cmake -S tools/vpaot -B build/vpaot -G Ninja && ninja -C build/vpaot
python3 tests/aot/run.py            # needs gcc/clang and binutils; native comparison on x86-64
./build/vpaot/vpaot --elf /bin/ls   # coverage report of any x86-64 ELF (or a PS4 SELF)
```

## License and credits

PolyForm Noncommercial 1.0.0 (see `LICENSE`). The translator decodes with [Zydis](https://github.com/zyantific/zydis) (MIT).
Written by Claude, building on the great work of shadPS4, bbport, FEX, Zydis, the N64Recomp and
XenonRecomp static recompilers, AstroQuest and pt-ipad; the repository owner contributed ideas on
how some things could be adapted.
