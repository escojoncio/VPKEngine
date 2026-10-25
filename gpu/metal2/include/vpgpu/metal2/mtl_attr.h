// Live Metal objects by creation site, to split MTLDevice.currentAllocatedSize (Objective-C++ only; header-only,
// one instance per image). Each creation site passes its static tag ("m2.tex.msaa"); the object goes into a weak
// table, so nothing hooks the release: the heartbeat (summary) walks the tables and adds what is still alive.
//   track        MTLBuffer / MTLTexture: summed by allocatedSize ("tracked"); >= 32 MiB logs gpu.bigalloc at once.
//   trackCount   views (share their base's memory), functions: counted, never summed.
//   trackObject  libraries, pipelines, archives (no allocatedSize): counted; the growth of currentAllocatedSize
//                measured around the creation is added up per tag (cumulative, approximate: "~").
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpgpu::metal2::attr {

enum class Kind : uint8_t { Resource, Count, Object };

struct Site {
  const char* tag;
  Kind kind;
  NSHashTable* live;
  uint64_t delta;  // Object: currentAllocatedSize growth seen around its creations (cumulative)
};

struct State {
  std::mutex mu;                       // sites, devices
  std::vector<Site> sites;
  std::unordered_map<const char*, size_t> byPtr;  // tag pointer -> index in sites (tags are literals)
  std::vector<id<MTLDevice>> devices;  // distinct devices seen (a second one would split the totals)
  std::mutex logMu;                    // log, bigWarned
  std::function<void(const std::string&)> log;
  int bigWarned = 0;
};

inline State& state() {
  static State* s = new State;  // leaked on purpose: guest threads may create objects during exit
  return *s;
}

inline void setLog(std::function<void(const std::string&)> log) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.logMu);
  s.log = std::move(log);
}

inline NSHashTable* newTable() {
  return [NSHashTable hashTableWithOptions:NSPointerFunctionsWeakMemory | NSPointerFunctionsObjectPointerPersonality];
}

// Pairs the tag's table with `s.sites`; call with `s.mu` held. Tags are string literals: the pointer is looked up
// first, strcmp only the first time a pointer is seen (two translation units may hold their own copy).
inline Site& siteLocked(State& s, const char* tag, Kind kind) {
  if (auto it = s.byPtr.find(tag); it != s.byPtr.end()) return s.sites[it->second];
  for (size_t i = 0; i < s.sites.size(); ++i)
    if (!std::strcmp(s.sites[i].tag, tag)) {
      s.byPtr[tag] = i;
      return s.sites[i];
    }
  s.sites.push_back(Site{tag, kind, newTable(), 0});
  s.byPtr[tag] = s.sites.size() - 1;
  return s.sites.back();
}

inline void noteDeviceLocked(State& s, id<MTLDevice> d) {
  if (!d || s.devices.size() >= 4) return;
  for (id<MTLDevice> x : s.devices)
    if (x == d) return;
  s.devices.push_back(d);
}

constexpr uint64_t kBigAlloc = 32ull << 20;
constexpr int kMaxBigWarnings = 200;

// One line per allocation >= 32 MiB (first 200 of the session): who, how big, what it is.
inline void warnBig(State& s, const char* tag, id<MTLResource> r, uint64_t bytes) {
  static const char* const kStorage[] = {"shared", "managed", "private", "memoryless"};
  const unsigned st = unsigned(r.storageMode);
  std::string label = r.label ? std::string(r.label.UTF8String) : std::string();
  for (char& c : label)
    if (c == ' ') c = '_';
  char b[384];
  if ([r conformsToProtocol:@protocol(MTLTexture)]) {
    id<MTLTexture> t = (id<MTLTexture>)r;
    std::snprintf(b, sizeof b,
                  "gpu.bigalloc tag=%s type=tex mib=%.1f dims=%lux%lux%lux%lu samples=%lu levels=%lu pf=%lu "
                  "storage=%s usage=0x%lx label=%s",
                  tag, double(bytes) / 1048576.0, (unsigned long)t.width, (unsigned long)t.height,
                  (unsigned long)t.depth, (unsigned long)t.arrayLength, (unsigned long)t.sampleCount,
                  (unsigned long)t.mipmapLevelCount, (unsigned long)t.pixelFormat, st < 4 ? kStorage[st] : "?",
                  (unsigned long)t.usage, label.c_str());
  } else {
    id<MTLBuffer> buf = (id<MTLBuffer>)r;
    std::snprintf(b, sizeof b, "gpu.bigalloc tag=%s type=buf mib=%.1f length=%lu storage=%s label=%s", tag,
                  double(bytes) / 1048576.0, (unsigned long)buf.length, st < 4 ? kStorage[st] : "?", label.c_str());
  }
  std::lock_guard<std::mutex> lock(s.logMu);
  if (s.bigWarned > kMaxBigWarnings) return;
  if (s.log) s.log(s.bigWarned == kMaxBigWarnings ? "gpu.bigalloc (limit of 200 reached, no more lines)" : b);
  ++s.bigWarned;
}

// Call after the object is final (label set). Returns `obj` so a creation expression can be wrapped.
inline id track(const char* tag, id obj) {
  if (!obj) return obj;
  id<MTLResource> r = obj;
  const uint64_t bytes = r.allocatedSize;
  State& s = state();
  {
    std::lock_guard<std::mutex> lock(s.mu);
    [siteLocked(s, tag, Kind::Resource).live addObject:obj];
    noteDeviceLocked(s, r.device);
  }
  if (bytes >= kBigAlloc) warnBig(s, tag, r, bytes);
  return obj;
}

inline id trackCount(const char* tag, id obj) {
  if (!obj) return obj;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  [siteLocked(s, tag, Kind::Count).live addObject:obj];
  return obj;
}

// currentAllocatedSize before the creation; only for the rare sites (libraries, pipelines, archives).
struct Probe {
  id<MTLDevice> dev;
  uint64_t before;
  explicit Probe(id<MTLDevice> d) : dev(d), before(d ? uint64_t(d.currentAllocatedSize) : 0) {}
};

inline id trackObject(const char* tag, id obj, const Probe& p) {
  if (!obj) return obj;
  const uint64_t now = p.dev ? uint64_t(p.dev.currentAllocatedSize) : 0;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  Site& x = siteLocked(s, tag, Kind::Object);
  [x.live addObject:obj];
  if (now > p.before) x.delta += now - p.before;
  noteDeviceLocked(s, p.dev);
  return obj;
}

// " attr_tracked=<bytes> attr_devs=N attr_top=tag:MiB/live,... attr_obj=tag:~MiB/live,... attr_cnt=tag:live,..."
// for the heartbeat: tracked = MTLBuffer/MTLTexture bytes still alive, top = the 12 largest tags, obj = pipelines /
// libraries with the allocation growth seen around their creation (~, cumulative), cnt = views / functions alive.
// With several devices, attr_dev_alloc lists each one's currentAllocatedSize (MiB). Any thread; the walk runs here,
// not at creation.
inline std::string summary() {
  struct Row {
    const char* tag;
    Kind kind;
    uint64_t bytes, live;
  };
  std::vector<Row> rows;
  std::vector<id<MTLDevice>> devices;
  uint64_t tracked = 0;
  @autoreleasepool {
    State& s = state();
    std::vector<NSArray*> objs;
    {
      std::lock_guard<std::mutex> lock(s.mu);  // copies only: the sums run outside the lock
      for (Site& x : s.sites) {
        rows.push_back(Row{x.tag, x.kind, x.delta, 0});
        NSArray* alive = x.live.allObjects;
        // Dead weak entries pile up under churn (per-copy staging buffers): rebuild the table with the living only.
        const NSUInteger held = x.live.count;
        if (held > 256 && held > 2 * alive.count) {
          NSHashTable* t = newTable();
          for (id o in alive) [t addObject:o];
          x.live = t;
        }
        objs.push_back(alive);
      }
      devices = s.devices;
    }
    for (size_t i = 0; i < rows.size(); ++i) {
      rows[i].live = objs[i].count;
      if (rows[i].kind != Kind::Resource) continue;
      for (id<MTLResource> r in objs[i]) rows[i].bytes += r.allocatedSize;
      tracked += rows[i].bytes;
    }
  }
  std::string out = " attr_tracked=" + std::to_string(tracked) + " attr_devs=" + std::to_string(devices.size());
  if (devices.size() > 1) {
    out += " attr_dev_alloc=";
    for (size_t i = 0; i < devices.size(); ++i)
      out += (i ? "/" : "") + std::to_string(uint64_t(devices[i].currentAllocatedSize) >> 20);
  }
  auto list = [&](const char* key, Kind kind, size_t max, bool byBytes) {
    std::vector<Row> v;
    for (const Row& r : rows)
      if (r.kind == kind && r.live) v.push_back(r);
    std::sort(v.begin(), v.end(), [&](const Row& a, const Row& b) {
      return byBytes ? a.bytes > b.bytes : a.live > b.live;
    });
    if (v.size() > max) v.resize(max);
    if (v.empty()) return;
    out += std::string(" ") + key + "=";
    char b[96];
    for (size_t i = 0; i < v.size(); ++i) {
      if (kind == Kind::Count) std::snprintf(b, sizeof b, "%s%s:%llu", i ? "," : "", v[i].tag, (unsigned long long)v[i].live);
      else
        std::snprintf(b, sizeof b, "%s%s:%s%.1fM/%llu", i ? "," : "", v[i].tag, kind == Kind::Object ? "~" : "",
                      double(v[i].bytes) / 1048576.0, (unsigned long long)v[i].live);
      out += b;
    }
  };
  list("attr_top", Kind::Resource, 12, true);
  list("attr_obj", Kind::Object, 6, true);
  list("attr_cnt", Kind::Count, 6, false);
  return out;
}

}  // namespace vpgpu::metal2::attr
