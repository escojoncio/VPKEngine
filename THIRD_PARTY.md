# Third-party components

VPKEngine is licensed under PolyForm Noncommercial 1.0.0 (see `LICENSE`). The components below are the only third-party
material in this repository or fetched by its build and CI; everything else is original VPKEngine code.

## In the repository

| Component | Used for | Location | License | Upstream | Modified? |
|---|---|---|---|---|---|
| Zydis (with its Zycore dependency) | x86-64 instruction decoding in the translator (`vpaot`, `vpconvert`) | `third_party/zydis` (git submodule, not copied); license copy in `docs/LICENSE-zydis` | MIT | https://github.com/zyantific/zydis | No, used as a pinned submodule |
| Berkeley SoftFloat 3e | Exact 80-bit x87 arithmetic in the runtime (`runtime/vp_x87.c`) | `runtime/softfloat/` (`COPYING.txt` holds the license) | BSD-3-Clause | http://www.jhauser.us/arithmetic/SoftFloat.html | Sources unchanged apart from one mechanical edit: the `#ifndef softfloat_X` guards test `VP_SF_INLINED_softfloat_X` and the matching `#define softfloat_X softfloat_X` lines are removed (affects `s_*.c` helpers); all symbols are renamed to `vp_sf_*` through `vp_softfloat_rename.h`. VPKEngine-written files in that directory: `platform.h`, `vp_softfloat.c`, `vp_softfloat_rename.h` |
| Khronos Vulkan-Headers v1.3.296 (subset) | Vulkan API types and prototypes for VPKEngine's own Vulkan driver (`gpu/vulkan`) | `gpu/vulkan/third_party/vulkan-headers/` (`LICENSE.md` there) | Apache-2.0 OR MIT | https://github.com/KhronosGroup/Vulkan-Headers | No, files copied unchanged (headers only; no Vulkan implementation code) |

## Fetched at build or CI time (not stored in the repository)

| Component | Used for | Where | License | Upstream |
|---|---|---|---|---|
| LLVM, clang, lld 21.1.0 | In-process compiler and Mach-O linker linked into `vpconvert` for visionOS | Downloaded and built by `tools/vpconvert/build-llvm-visionos.sh` (CI: `vpconvert-visionos.yml`); only headers and static libraries are linked | Apache-2.0 with LLVM exceptions | https://github.com/llvm/llvm-project |
| apple-codesign (`rcodesign`) 0.29.0 | Independent verification of the code signatures the runtime writes (test only) | Downloaded in `.github/workflows/aot-tests.yml`; not shipped | MPL-2.0 | https://github.com/indygreg/apple-platform-rs |
| OpenSSL (libcrypto) | Test-side signing and verification; PKCS#12 ciphers on non-Apple hosts (`runtime/vp_pkcs12.c`) | System package, linked dynamically | Apache-2.0 (3.x) | https://www.openssl.org |
| CommonCrypto / Security.framework | PKCS#12 ciphers and private-key operations on Apple systems | System frameworks | Apple system software | n/a |
| GitHub Actions (`actions/checkout`, `actions/cache`, `actions/upload-artifact`, `actions/download-artifact`, `softprops/action-gh-release`) | CI plumbing | `.github/workflows/` | MIT | https://github.com/actions |
| Build tools (clang, lld, binutils, gcc cross tools, ninja, cmake, xcodegen) | Building and testing | Installed by CI via apt / brew | Their own licenses; not part of the repository | n/a |

Standard public constants and formats (for example the SHA-256 round constants in `runtime/vp_codesign.c`,
FIPS 180-4; PKCS#12, CMS and the Mach-O code-signature layout, implemented from their published
specifications) are used from the specifications, not copied from another implementation.

Everything else in this repository is original VPKEngine code.
