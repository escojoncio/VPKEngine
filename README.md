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

Status: the translator covers what the PS4's CPU (AMD Jaguar) runs: integer and control flow,
SSE through SSE4.2 (including the string compares and CRC32), SSE4a, AVX at 128 and 256 bits,
F16C, AES-NI, PCLMULQDQ, BMI1, a bit-exact x87 FPU (80-bit, via Berkeley SoftFloat), and locked
operations, which stay atomic even on misaligned addresses. It also handles jump tables, C++
exceptions through the game's own unwinder, and context switches (fibers, coroutines, `longjmp`).

- **Verification:** every case is compared bit for bit with the hardware, on Intel locally and on
  AMD in CI, and replayed on ARM.
- **Coverage on real binaries:**
  - python3 (1.5M instructions) and libstdc++ (1M) translate at 100.00 %.
  - libstdc++ and libc, translated whole, compile with no error.
- **Speed:** translated code runs at 1.05–1.13× the time of native x86 (clang -O2).
- **AstroVisionPro:** `integrations/shadps4/` plugs this into AstroVisionPro's shadPS4 in place
  of FEX, with no JIT. What a whole game still needs is listed in [docs/PLAN.md](docs/PLAN.md) and
  `bitacora.md`.

No game data is included or needed to build or test.

## Build and test

```sh
git clone --recursive <this repository> && cd VPEngine
cmake -S tools/vpaot -B build/vpaot -G Ninja && ninja -C build/vpaot
python3 tests/aot/run.py            # needs gcc/clang and binutils; native comparison on x86-64
./build/vpaot/vpaot --elf /bin/ls   # coverage report of any x86-64 ELF (or a PS4 SELF)
```

## License and credits

PolyForm Noncommercial 1.0.0 (see `LICENSE`). The translator decodes with [Zydis](https://github.com/zyantific/zydis) (MIT);
the x87 FPU computes with [Berkeley SoftFloat 3e](http://www.jhauser.us/arithmetic/SoftFloat.html)
(BSD-3-Clause, `runtime/softfloat/COPYING.txt`).
Written by Claude, building on the great work of shadPS4, bbport, FEX, Zydis, SoftFloat, the N64Recomp and
XenonRecomp static recompilers, AstroQuest and pt-ipad; the repository owner contributed ideas on
how some things could be adapted.
