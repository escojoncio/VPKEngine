// vpimgdiff: compares two vpreplay dump directories (index.json + PNG / raw files, matched by file name) and reports
// the first dump that diverges, in the order of A's index.json, with the call and the pipeline that produced it.
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "vpimage.h"
#include "vpjson.h"

namespace vpimgdiff {

struct Options {
  uint32_t maxDiff = 0;  // largest per-channel difference allowed, in 8-bit units (a 16-bit dump allows 257x as many)
  double minPsnr = 0;    // 0 = not checked; else a dump below this PSNR (dB) diverges
  bool all = false;      // list every divergent dump, not only the first
};

struct Counts {
  int compared = 0, divergent = 0, missingInB = 0, extraInB = 0;
};

inline bool readFile(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

inline bool loadIndex(const std::string& dir, vpjson::Value& idx, std::string& err) {
  std::vector<uint8_t> raw;
  if (!readFile(dir + "/index.json", raw)) return err = dir + "/index.json: cannot read", false;
  const std::string text(raw.begin(), raw.end());
  vpjson::Parser p(text);
  if (!p.parse(idx) || idx.type != vpjson::Value::Object || !idx.get("dumps") || idx.get("dumps")->type != vpjson::Value::Array)
    return err = dir + "/index.json: malformed", false;
  return true;
}

// A name from an index.json becomes a path under the directory: nothing that leaves it.
inline bool safeName(const std::string& n) {
  return !n.empty() && n.find('/') == std::string::npos && n.find('\\') == std::string::npos && n.find("..") == std::string::npos;
}

// 0: no dump diverges, 1: at least one does (or none was compared at all), 2: an index could not be read.
inline int compareDirs(const std::string& dirA, const std::string& dirB, const Options& opt, std::FILE* out, Counts* counts = nullptr) {
  vpjson::Value a, b;
  std::string err;
  if (!loadIndex(dirA, a, err) || !loadIndex(dirB, b, err)) {
    std::fprintf(out, "vpimgdiff: %s\n", err.c_str());
    return 2;
  }
  for (const vpjson::Value* idx : {&a, &b})
    for (const auto& d : idx->get("dumps")->arr)
      if (!safeName(d.text("file"))) {
        std::fprintf(out, "vpimgdiff: index.json names a file that is not a plain file name: \"%s\"\n", d.text("file").c_str());
        return 2;
      }
  std::map<std::string, const vpjson::Value*> inB;
  for (const auto& d : b.get("dumps")->arr) inB[d.text("file")] = &d;
  Counts c;
  std::map<std::string, bool> seen;
  bool first = true;
  auto describe = [&](const vpjson::Value& d) {
    char buf[320];
    std::snprintf(buf, sizeof buf, "call=%s call_index=%.0f pipeline=%.0f", d.text("call").c_str(), d.number("call_index"), d.number("pipeline"));
    return std::string(buf);
  };
  auto report = [&](const std::string& what, const vpjson::Value& da, const vpjson::Value* db) {
    ++c.divergent;
    if (!first && !opt.all) return;
    std::fprintf(out, "%s %s: %s  [A %s", first ? "FIRST DIVERGENCE" : "divergence", da.text("file").c_str(), what.c_str(), describe(da).c_str());
    if (db) std::fprintf(out, " | B %s", describe(*db).c_str());
    std::fprintf(out, "]\n");
    first = false;
  };
  for (const auto& da : a.get("dumps")->arr) {
    const std::string name = da.text("file");
    seen[name] = true;
    auto it = inB.find(name);
    if (it == inB.end()) {
      ++c.missingInB;
      report("missing in B", da, nullptr);
      continue;
    }
    const vpjson::Value& db = *it->second;
    ++c.compared;
    std::vector<uint8_t> fa, fb;
    if (!readFile(dirA + "/" + name, fa) || !readFile(dirB + "/" + name, fb)) {
      report("file unreadable", da, &db);
      continue;
    }
    if (da.text("kind") == "raw" || db.text("kind") == "raw") {
      if (fa != fb) report("raw bytes differ", da, &db);
      continue;
    }
    vpimg::Image ia, ib;
    if (!vpimg::pngDecode(fa, ia, &err) || !vpimg::pngDecode(fb, ib, &err)) {
      report("PNG unreadable: " + err, da, &db);
      continue;
    }
    const vpimg::DiffResult r = vpimg::diffImages(ia, ib);
    char buf[256];
    if (r.sizeMismatch) {
      std::snprintf(buf, sizeof buf, "image shape differs (%ux%u/%u bit vs %ux%u/%u bit)", ia.w, ia.h, ia.bits, ib.w, ib.h, ib.bits);
      report(buf, da, &db);
      continue;
    }
    const uint32_t allowed = opt.maxDiff * (ia.bits == 16 ? 257u : 1u);
    if (r.maxOf() > allowed || (opt.minPsnr > 0 && r.psnr < opt.minPsnr)) {
      char ch[4][16];
      for (int k = 0; k < 4; ++k) std::snprintf(ch[k], sizeof ch[k], std::isinf(r.psnrChannel[k]) ? "inf" : "%.1f", r.psnrChannel[k]);
      std::snprintf(buf, sizeof buf, "psnr=%.2f dB (rgba %s,%s,%s,%s) max_diff rgba=%u,%u,%u,%u (%u-bit)", r.psnr, ch[0], ch[1], ch[2], ch[3],
                    r.maxDiff[0], r.maxDiff[1], r.maxDiff[2], r.maxDiff[3], ia.bits);
      report(buf, da, &db);
    }
  }
  for (const auto& d : b.get("dumps")->arr)
    if (!seen.count(d.text("file"))) ++c.extraInB;
  std::fprintf(out, "compared=%d divergent=%d missing_in_b=%d extra_in_b=%d\n", c.compared, c.divergent, c.missingInB, c.extraInB);
  if (counts) *counts = c;
  // A comparison of nothing proves nothing (two empty replays, or no name in common): it fails.
  if (!c.compared) std::fprintf(out, "NOTHING COMPARED: no dump is in both directories\n");
  return c.divergent || c.extraInB || !c.compared ? 1 : 0;
}

}  // namespace vpimgdiff
