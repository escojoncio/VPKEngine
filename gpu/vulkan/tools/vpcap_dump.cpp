// vpcap_dump <file.vpcap> [-l] : header, META / END text, per-call-name totals; -l also lists every call.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "vpvk/vpcap.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: vpcap_dump <file.vpcap> [-l]\n");
    return 2;
  }
  const bool list = argc > 2 && !std::strcmp(argv[2], "-l");
  vpcap::Reader r;
  std::string err;
  if (!r.open(argv[1], &err)) {
    std::fprintf(stderr, "vpcap_dump: %s: %s\n", argv[1], err.c_str());
    return 1;
  }
  std::map<std::string, uint64_t> state, frame;
  uint64_t stateCalls = 0, frameCalls = 0, frames = 0, memBytes = 0;
  vpcap::Call c;
  while (r.next(c)) {
    const bool st = c.section == vpcap::Section::State;
    ++(st ? stateCalls : frameCalls);
    ++(st ? state : frame)[vpcap::callName(c.id)];
    if (c.id == vpcap::id::vpcapFrameEnd) ++frames;
    vpcap::MemoryData md;
    if (vpcap::decodeMemoryData(c, md)) memBytes += md.size;
    if (!list) continue;
    std::printf("%6llu %-5s f%-3u t%-2u %-34s %u bytes", static_cast<unsigned long long>(c.index), st ? "state" : "frame",
                c.frame, c.thread, vpcap::callName(c.id), c.size);
    if (md.data) std::printf("  mem=%llu off=%llu len=%llu", static_cast<unsigned long long>(md.memory),
                             static_cast<unsigned long long>(md.offset), static_cast<unsigned long long>(md.size));
    std::printf("\n");
  }
  if (!r.error().empty()) {
    std::fprintf(stderr, "vpcap_dump: %s\n", r.error().c_str());
    return 1;
  }
  std::printf("version=%u\n%s", r.version(), r.meta().c_str());
  std::printf("state calls=%llu frame calls=%llu frames=%llu memory_bytes=%llu\n", static_cast<unsigned long long>(stateCalls),
              static_cast<unsigned long long>(frameCalls), static_cast<unsigned long long>(frames),
              static_cast<unsigned long long>(memBytes));
  for (const auto& [n, k] : state) std::printf("  state %-34s %llu\n", n.c_str(), static_cast<unsigned long long>(k));
  for (const auto& [n, k] : frame) std::printf("  frame %-34s %llu\n", n.c_str(), static_cast<unsigned long long>(k));
  return 0;
}
