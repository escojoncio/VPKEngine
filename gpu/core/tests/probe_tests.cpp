// gpu/core probe tests: counters, distributions, line format, CSV, summary, timed mutex.
// Usage: vpgpu_probe_tests [group]  (no group = all).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "vpgpu/core/probe.h"

using namespace vpgpu::probe;

namespace {

int failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                   \
    }                                                               \
  } while (0)
#define CHECK_EQ(a, b)                                                                   \
  do {                                                                                   \
    const auto va = (a);                                                                 \
    const auto vb = (b);                                                                 \
    if (!(va == vb)) {                                                                   \
      std::printf("  FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, \
                  (long long)va, (long long)vb);                                         \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

#define CHECK_STR(a, b)                                                                   \
  do {                                                                                    \
    const std::string va = (a);                                                           \
    const std::string vb = (b);                                                           \
    if (va != vb) {                                                                       \
      std::printf("  FAIL %s:%d: %s == %s (\"%s\" vs \"%s\")\n", __FILE__, __LINE__, #a, #b, \
                  va.c_str(), vb.c_str());                                                \
      ++failures;                                                                         \
    }                                                                                     \
  } while (0)

// The value after "key=" in a line (empty if the key is absent).
std::string field(const std::string& line, const std::string& key) {
  const std::string needle = " " + key + "=";
  const size_t at = line.find(needle);
  if (at == std::string::npos) return {};
  const size_t from = at + needle.size();
  const size_t to = line.find(' ', from);
  return line.substr(from, to == std::string::npos ? std::string::npos : to - from);
}

std::string line(const std::vector<std::string>& lines, const std::string& family) {
  for (const std::string& l : lines)
    if (l.rfind("VPKENGINE_PROBE " + family + " ", 0) == 0) return l;
  return {};
}

void testBusyUnion() {
  Timer t;
  BusyUnion u(t);
  u.add(100, 200);  // 100
  u.add(150, 300);  // overlaps: +100
  u.add(250, 280);  // inside: +0
  u.add(400, 450);  // gap: +50
  CHECK_EQ(t.ns(), 250u);
  CHECK_EQ(t.calls(), 4u);
  // Out of completion order (two queues, or handlers that overtake each other): what an earlier interval left idle still counts.
  Timer late;
  BusyUnion v(late);
  v.add(2, 5);        // 3
  v.add(0, 10);       // the two ends around it: +7
  v.add(600, 1000);   // 400
  v.add(100, 500);    // before it, in what was idle: +400
  v.add(450, 550);    // +50 (500..550)
  v.add(90, 1000);    // +10 (90..100) +50 (550..600)
  CHECK_EQ(late.ns(), 920u);
  CHECK_EQ(late.calls(), 6u);
}

void testHist() {
  // Small values are exact, buckets grow with the value, and a bucket's middle is inside it.
  for (uint64_t v = 0; v < 16; ++v) CHECK_EQ(Hist::bucketOf(v), unsigned(v));
  unsigned last = 0;
  for (uint64_t v = 1; v < (1ull << 40); v = v * 3 / 2 + 1) {
    const unsigned b = Hist::bucketOf(v);
    CHECK(b >= last);
    CHECK(b < Hist::kBuckets);
    last = b;
    const uint64_t mid = Hist::bucketMid(b);
    CHECK(mid <= v * 2 && mid * 2 >= v);
  }
  CHECK(Hist::bucketOf(~0ull) < Hist::kBuckets);

  Hist h;
  for (uint64_t v = 1; v <= 100; ++v) h.add(v);
  h.add(5000);
  Hist::Snapshot s = h.take();
  CHECK_EQ(s.n, uint64_t{101});
  CHECK_EQ(s.max, uint64_t{5000});
  CHECK(s.p50 >= 44 && s.p50 <= 58);
  CHECK(s.p90 >= 80 && s.p90 <= 100);
  CHECK(s.p99 >= 90 && s.p99 <= 5000);
  CHECK(s.p50 <= s.p90 && s.p90 <= s.p99 && s.p99 <= s.max);
  CHECK_EQ(s.sum, uint64_t{5050 + 5000});
  s = h.take();  // taken: the interval starts again
  CHECK_EQ(s.n, uint64_t{0});
  CHECK_EQ(s.max, uint64_t{0});
  CHECK_EQ(h.total(), uint64_t{101});
}

void testLines() {
  Registry r;
  Counter& draws = r.counter("RATES", "draws");
  Gauge& mb = r.gauge("MEM", "footprint_mb");
  Timer& tm = r.timer("TIME", "bind");
  Hist& hi = r.hist("TIME", "submit_us");
  r.info("MEM", "ledger", "resident=12MiB n=3");
  draws.add(100);
  mb.set(2791.5);
  tm.add(2'000'000);  // 2 ms
  tm.add(6'000'000);  // 6 ms
  for (int i = 1; i <= 10; ++i) hi.add(uint64_t(i) * 1000);
  std::vector<std::string> a = r.sample(2.0);
  CHECK_EQ(a.size(), size_t{3});  // families in order of first use: RATES, MEM, TIME
  const std::string rates = line(a, "RATES"), mem = line(a, "MEM"), time = line(a, "TIME");
  CHECK(rates.rfind("VPKENGINE_PROBE RATES i=1 t=2.0 dt=2.00", 0) == 0);
  CHECK_STR(field(rates, "draws"), std::string("100"));
  CHECK_STR(field(rates, "draws_d"), std::string("100"));
  CHECK_STR(field(mem, "footprint_mb"), std::string("2791.5"));
  CHECK_STR(field(mem, "ledger"), std::string("resident=12MiB_n=3"));  // no spaces in a value
  CHECK_STR(field(time, "bind_n_d"), std::string("2"));
  CHECK_STR(field(time, "bind_ms_s"), std::string("4"));  // 8 ms over 2 s
  CHECK_STR(field(time, "bind_max_us"), std::string("6000"));
  CHECK_STR(field(time, "submit_us_n_d"), std::string("10"));
  CHECK_STR(field(time, "submit_us_max"), std::string("10000"));
  CHECK_STR(field(time, "submit_us_ms_s"), std::string("27.5"));  // 55 ms over 2 s
  const long p50 = std::stol(field(time, "submit_us_p50")), p99 = std::stol(field(time, "submit_us_p99"));
  CHECK(p50 >= 4000 && p50 <= 6500);
  CHECK(p99 >= 9000 && p99 <= 10000);

  draws.add(50);
  std::vector<std::string> b = r.sample(4.0);
  const std::string rates2 = line(b, "RATES");
  CHECK_STR(field(rates2, "draws"), std::string("150"));  // cumulative
  CHECK_STR(field(rates2, "draws_d"), std::string("50"));  // delta of the interval
  CHECK_STR(field(line(b, "TIME"), "bind_n_d"), std::string("0"));
  CHECK_STR(field(line(b, "TIME"), "bind_n"), std::string("2"));
  // A logger that drops repeated messages must never see the same line twice, even when nothing changed.
  std::vector<std::string> c = r.sample(6.0), d = r.sample(8.0);
  CHECK(line(c, "RATES") != line(d, "RATES"));
  CHECK(line(c, "MEM") != line(d, "MEM"));
  CHECK_EQ(r.samples(), uint64_t{4});
}

void testNamesAndNumbers() {
  CHECK_STR(Registry::sanitize("Staging memory limit reached: copy source refused"), std::string("staging_memory_limit_reached_copy_source_refused"));
  CHECK_STR(Registry::sanitize("  ps: texture unavailable "), std::string("ps_texture_unavailable"));
  CHECK_STR(Registry::sanitize("!!!"), std::string("_"));
  CHECK(Registry::sanitize(std::string(200, 'a')).size() <= 48);
  CHECK_STR(Registry::number(3.0), std::string("3"));
  CHECK_STR(Registry::number(0.1250), std::string("0.125"));
  CHECK_STR(Registry::number(1234.5678), std::string("1234.568"));
}

std::vector<std::vector<std::string>> readCsv(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::vector<std::string>> rows;
  std::string l;
  while (std::getline(f, l)) {
    std::vector<std::string> cells(1);
    for (char ch : l) {
      if (ch == ',') cells.emplace_back();
      else cells.back() += ch;
    }
    rows.push_back(cells);
  }
  return rows;
}

void testCsv() {
  const std::string path = "probe_test_csv.txt";
  Registry r;
  CHECK(r.openCsv(path));
  Counter& c = r.counter("A", "x");
  c.add(7);
  r.sample(1.0);
  r.gauge("B", "y").set(2.5);  // a metric that appears later: the file is written again with it
  c.add(3);
  r.sample(2.0);
  r.timer("B", "z").add(1'000'000);
  r.sample(3.0);
  r.openCsv("");  // closed
  const auto rows = readCsv(path);
  CHECK_EQ(rows.size(), size_t{4});  // header + three intervals
  if (rows.size() == 4) {
    CHECK_STR(rows[0][0], std::string("i"));
    CHECK_STR(rows[0][1], std::string("t"));
    CHECK_STR(rows[0][2], std::string("dt"));
    CHECK_STR(rows[0][3], std::string("A.x"));
    CHECK_STR(rows[0][4], std::string("A.x_d"));
    CHECK_STR(rows[0][5], std::string("B.y"));
    CHECK_STR(rows[0][6], std::string("B.z_n_d"));
    for (const auto& row : rows) CHECK_EQ(row.size(), rows[0].size());  // one column per metric, every row complete
    CHECK_STR(rows[1][3], std::string("7"));
    CHECK_STR(rows[1][5], std::string(""));  // not measured yet
    CHECK_STR(rows[2][3], std::string("10"));
    CHECK_STR(rows[2][4], std::string("3"));
    CHECK_STR(rows[2][5], std::string("2.5"));
    CHECK_STR(rows[3][6], std::string("1"));
  }
  std::remove(path.c_str());
  Registry bad;
  CHECK(!bad.openCsv("/nonexistent-dir/probe.txt"));
}

void testSummary() {
  Registry r;
  CHECK(r.summary().size() == 1 && r.summary()[0].find("empty") != std::string::npos);
  Counter& c = r.counter("F", "draws");
  Gauge& g = r.gauge("F", "mb");
  Hist& h = r.hist("F", "us");
  c.add(100);
  g.set(10);
  h.add(100);
  r.sample(1.0);
  c.add(300);
  g.set(30);
  h.add(900);
  r.sample(2.0);
  g.set(20);
  r.sample(3.0);
  const std::vector<std::string> s = r.summary();
  const std::string f = line(s, "SUMMARY F");
  CHECK(!f.empty());
  CHECK_STR(field(f, "samples"), std::string("3"));
  CHECK_STR(field(f, "draws_total"), std::string("400"));
  CHECK_STR(field(f, "draws_rate_peak"), std::string("300"));
  CHECK_STR(field(f, "mb_peak"), std::string("30"));
  CHECK_STR(field(f, "mb_avg"), std::string("20"));
  CHECK_STR(field(f, "mb_last"), std::string("20"));
  CHECK_STR(field(f, "us_max_peak"), std::string("900"));
  const std::vector<std::string> again = r.summary();
  CHECK(line(again, "SUMMARY F") != f);  // numbered
}

void testScopedAndMutex() {
  Registry& g = registry();
  g.reset();
  setEnabled(false);
  Timer& t = timer("TIME", "off");
  {
    Scoped s(t);
  }
  CHECK_EQ(t.calls(), uint64_t{0});  // off: nothing is read or counted
  setEnabled(true);
  {
    Scoped s(t);
  }
  CHECK_EQ(t.calls(), uint64_t{1});

  TimedMutex<std::mutex> m("test_mutex");
  { std::lock_guard l(m); }
  CHECK_EQ(counter("LOCK", "test_mutex_acq").get(), uint64_t{1});
  CHECK_EQ(timer("LOCK", "test_mutex_wait").calls(), uint64_t{0});  // not contended
  m.lock();
  std::thread other([&] {
    threadTag = "t";  // (a thread with a name: its waits are also filed apart)
    std::lock_guard l(m);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  m.unlock();
  other.join();
  CHECK_EQ(counter("LOCK", "test_mutex_acq").get(), uint64_t{3});
  CHECK_EQ(timer("LOCK", "test_mutex_wait").calls(), uint64_t{1});  // the second thread waited
  CHECK_EQ(timer("LOCK", "test_mutex_wait_t").calls(), uint64_t{1});
  CHECK(timer("LOCK", "test_mutex_wait").ns() >= 10'000'000);
  TimedMutex<std::recursive_mutex> rm("test_recursive");
  rm.lock();
  rm.lock();  // the owner again: not a wait
  rm.unlock();
  rm.unlock();
  CHECK_EQ(timer("LOCK", "test_recursive_wait").calls(), uint64_t{0});
  setEnabled(false);
  g.reset();
}

void testBigAlloc() {
  registry().reset();
  setEnabled(false);
  AllocToken off = bigAllocBegin("tex_color", 64ull << 20, 0);
  CHECK(!off.active);
  CHECK_EQ(counter("ALLOC", "tex_color_n").get(), uint64_t{0});  // off: not even counted
  setEnabled(true);
  AllocToken small = bigAllocBegin("buf_own", 4096, 100);
  CHECK(!small.active);  // below the threshold: counted, no footprint read
  CHECK_EQ(counter("ALLOC", "buf_own_n").get(), uint64_t{1});
  CHECK_EQ(counter("ALLOC", "buf_own_bytes").get(), uint64_t{4096});
  AllocToken big = bigAllocBegin("tex_color", 64ull << 20, 1000);
  CHECK(big.active);
  bigAllocEnd(big, 1000 + (64ull << 20));
  CHECK_EQ(counter("ALLOC", "tex_color_big_n").get(), uint64_t{1});
  CHECK_EQ(counter("ALLOC", "tex_color_big_bytes").get(), uint64_t{64ull << 20});
  CHECK(gauge("ALLOC", "tex_color_big_device_mb").get() > 63.9 && gauge("ALLOC", "tex_color_big_device_mb").get() < 64.1);
  bigAllocEnd(small, 0);  // an inactive token does nothing
  CHECK_EQ(counter("ALLOC", "buf_own_big_n").get(), uint64_t{0});
  setEnabled(false);
  registry().reset();
}

}  // namespace

// Long names keep a hash suffix; lines are cut at ~2 KB; the compact log carries only what moved; a timer whose publisher
// does not know the longest call says so; the sampled timer counts every call.
void testLog() {
  const std::string a = Registry::sanitize("resolve dropped: " + std::string(60, 'x') + " one");
  const std::string b = Registry::sanitize("resolve dropped: " + std::string(60, 'x') + " two");
  CHECK(a.size() <= 48 && b.size() <= 48);
  CHECK(a != b);
  CHECK_STR(a, Registry::sanitize("resolve dropped: " + std::string(60, 'x') + " one"));  // deterministic
  CHECK_STR(Registry::sanitize("short name"), std::string("short_name"));

  Registry r;
  std::vector<Counter*> many;
  for (int i = 0; i < 400; ++i) many.push_back(&r.counter("BIG", "counter_number_" + std::to_string(i)));
  Counter& moving = r.counter("BIG", "moving");
  Gauge& still = r.gauge("BIG", "still");
  still.set(7);
  Timer& unknown = r.timer("BIG", "published");
  unknown.addMs(10, 4);
  for (Counter* c : many) c->add(1);
  std::vector<std::string> compact;
  std::vector<std::string> full = r.sample(1.0, &compact);
  CHECK(full.size() > 1);  // 400 counters do not fit in one line
  size_t parts = 0;
  for (const std::string& l : full) {
    CHECK(l.size() <= 2100);
    if (l.rfind("VPKENGINE_PROBE BIG ", 0) == 0) ++parts;
  }
  CHECK_EQ(parts, full.size());
  CHECK(full.front().find(" part=1/" + std::to_string(full.size())) != std::string::npos);
  CHECK_EQ(compact.size(), full.size());  // first interval: everything
  std::string joined;
  for (const std::string& l : full) joined += l;
  CHECK(joined.find(" moving=0 moving_d=0") != std::string::npos);
  CHECK(joined.find(" published_max_us=-") != std::string::npos);  // unknown, not 0

  moving.add(5);
  std::vector<std::string> compact2;
  std::vector<std::string> full2 = r.sample(2.0, &compact2);
  CHECK(full2.size() == full.size());  // the full lines are still complete
  CHECK_EQ(compact2.size(), size_t{1});
  CHECK(compact2[0].find(" moving=5 moving_d=5") != std::string::npos);
  CHECK(compact2[0].find("counter_number_3=") == std::string::npos);  // no delta: left out of the log
  CHECK(compact2[0].find(" still=") == std::string::npos);            // unchanged gauge
  CHECK(compact2[0].size() < 400);
  // Every kFullEvery-th interval is complete again.
  std::vector<std::string> last;
  for (uint64_t i = 3; i <= Registry::kFullEvery + 1; ++i) r.sample(double(i), &last);
  CHECK_EQ(last.size(), full.size());

  setEnabled(true);
  Registry s;
  Timer& hot = s.timer("T", "hot");
  for (int i = 0; i < 160; ++i) {
    ScopedSampled scope(hot);
  }
  CHECK_EQ(hot.calls(), uint64_t{160});
  setEnabled(false);
  {
    ScopedSampled scope(hot);
  }
  CHECK_EQ(hot.calls(), uint64_t{160});  // off: nothing is counted
}

int main(int argc, char** argv) {
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"hist", testHist}, {"busy_union", testBusyUnion}, {"lines", testLines}, {"names", testNamesAndNumbers}, {"csv", testCsv},
      {"summary", testSummary}, {"mutex", testScopedAndMutex}, {"bigalloc", testBigAlloc}, {"log", testLog}};
  bool ran = false;
  for (const auto& [name, fn] : groups) {
    if (argc > 1 && std::strcmp(argv[1], name) != 0) continue;
    std::printf("[%s]\n", name);
    fn();
    ran = true;
  }
  if (!ran) std::printf("unknown group\n");
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures || !ran ? 1 : 0;
}
