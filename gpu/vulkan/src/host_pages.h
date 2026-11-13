// Page-granular host memory that a no-copy Metal buffer may wrap (BufferDesc::guestMemory).
// MTLSimDevice (visionOS / iOS simulator) shares such memory through xpc_shmem_create, which traps (_xpc_api_misuse) on
// a block carved out of the malloc heap; a mapping of its own (mmap) is always accepted. Every block is 16 KiB aligned
// (Apple page) and a multiple of 16 KiB long, whatever the host's page size; pageFree takes the length pageAlloc was given.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#define VPVK_PAGE_MMAP 1
#endif

namespace vpvk {

constexpr uint64_t kHostPageBytes = 16384;

inline uint64_t pageRound(uint64_t bytes) {
  return (bytes + (kHostPageBytes - 1)) & ~(kHostPageBytes - 1);
}

// `tagged`: Apple only, VM tag 241 so the region map names the block (resident = staging and buffer contents).
inline uint8_t* pageAlloc(uint64_t bytes, bool tagged = false) {
  bytes = pageRound(bytes ? bytes : 1);
#ifdef VPVK_PAGE_MMAP
#ifdef __APPLE__
  const int tag = tagged ? int(241u << 24) : -1;  // VM_MAKE_TAG: Darwin reads an anonymous mapping's tag from the fd argument
#else
  (void)tagged;
  const int tag = -1;
#endif
  // 16 KiB aligned whatever the system's page size (4 KiB hosts): map one extra unit and give the ends back.
  void* p = mmap(nullptr, bytes + kHostPageBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, tag, 0);
  if (p == MAP_FAILED) return nullptr;
  uint8_t* raw = static_cast<uint8_t*>(p);
  uint8_t* out = raw + (kHostPageBytes - reinterpret_cast<uintptr_t>(raw) % kHostPageBytes) % kHostPageBytes;
  if (out > raw) munmap(raw, size_t(out - raw));
  if (const size_t tail = size_t((raw + bytes + kHostPageBytes) - (out + bytes))) munmap(out + bytes, tail);
  return out;
#else
  (void)tagged;
  return static_cast<uint8_t*>(std::aligned_alloc(kHostPageBytes, bytes));
#endif
}

inline void pageFree(uint8_t* p, uint64_t bytes) {
  if (!p) return;
#ifdef VPVK_PAGE_MMAP
  munmap(p, pageRound(bytes ? bytes : 1));
#else
  (void)bytes;
  std::free(p);
#endif
}

}  // namespace vpvk
