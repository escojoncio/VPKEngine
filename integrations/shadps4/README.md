# VPEngine as the guest CPU of AstroVisionPro's shadPS4

`aot_guest_engine.cpp` implements the interface of shadPS4's `src/core/fex/fex_guest_engine.h`
(`Core::Fex::GuestEngine` and the free functions declared there) with VPEngine's ahead-of-time
translated code. Everything else in shadPS4 (veneers, HLE bridge, fibers, Orbis signals) keeps
running its FEX code paths unchanged. There is no JIT, so the app needs no debugger attached and no
executable memory.

## 1. Translate the game (on the PC, once per game and per game update)

```
.\translate_game.ps1 -Game "A:\Games\CUSA03173" -Out "C:\vpengine-out" -Vpaot .\vpaot.exe
```

The script (in `tools/scripts/`, with a `.sh` twin) runs `vpaot --elf <module> --pic --module <name>
--split 4000` on `eboot.bin` and on every `sce_module\*.prx|*.sprx`. It then writes
`vpengine_registry.c`, which links every module in through `vpaot --registry`.

Each module also gets `<name>.json`. Its `supported_fraction` and `unsupported_by_mnemonic` show
what the translator does not cover yet.

> The C files are derived from the game. Keep them out of public repositories: build locally, or
> from a private repository or artifact.

## 2. Build shadPS4 with it

Apply `astrovisionpro.patch` to AstroVisionPro (`git apply`). It adds the option
`ENABLE_VPENGINE_GUEST_CPU`, which is off by default, so the FEX build is unchanged. Then configure:

```
-DENABLE_VPENGINE_GUEST_CPU=ON -DVPENGINE_DIR=<VPEngine checkout> -DVPENGINE_GAME_DIR=C:\vpengine-out
```

With the option on:

- `src/core/fex/fex_guest_engine.cpp` and the FEXCore libraries are replaced by
  `aot_guest_engine.cpp`, `runtime/vp_host.c` and the translated C.
- `vpengine.cmake` compiles that C with `-O2 -frounding-math`. `-frounding-math` is required:
  MXCSR rounding is mapped to the host's floating-point environment.
- `SHADPS4_ENABLE_FEX_GUEST_CPU` stays defined, so the rest of shadPS4 keeps its FEX paths.

Without `VPENGINE_GAME_DIR` the engine still builds, with no game translated.

## 3. Run, and close the gaps

The engine finds each module's translation by code fingerprint the first time the module runs,
wherever shadPS4 loaded it. A module whose bytes differ from what was translated never attaches.

Code that static analysis could not find is reached only through pointers it did not see. Each
such entry is logged once, as `module+0xOFFSET`, to `Documents/vpengine_missing.txt` in the app
container (visible in the Files app), or to `$VPENGINE_MISSING_LOG`. To close those gaps:

1. Copy that file to the PC.
2. Run `translate_game.ps1 ... -Missing vpengine_missing.txt` (`vpaot --roots` takes each module's
   own lines).
3. Rebuild.

## Tests

`tests/shadps4/` drives this file the way shadPS4 does, and runs in CI on x86-64 and ARM64. It
covers:

- shadPS4's interface headers, copied unchanged.
- Veneers written at run time.
- Two translated modules attached lazily at addresses other than their link addresses.
- Nested `CallGuest`, and guest threads with their own TLS.
- An HLE call with float arguments and result, during which an Orbis signal is delivered.
- An HLE failure after a callback.
- Seven engines created one after another.
- A static library that pulls in the modules through the registry.
