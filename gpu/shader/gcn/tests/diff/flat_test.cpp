// FLAT lowering: programs of FLAT instructions (hand-encoded) lifted as a compute shader and run on the VPIR
// interpreter with a guest memory map (the GuestMap binding) in front of "device" memory. Each program reads
// operands from user SGPRs s0..s7, runs, and the test compares the device memory.
//
//   s0:s1  guest address of the source        s2:s3  guest address of the destination
//   s4..s7 data registers v6..v9 of the program
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <array>
#include "vpir/ir.h"
#include "vpir/text.h"
#include "vpgcn/lift.h"
#include "vpir_eval.h"

namespace {

constexpr uint64_t kGuestBase = 0x200000000ull;  // the guest region (hi = 2)
constexpr uint32_t kGuestSize = 0x1000;
constexpr uint32_t kDevHi = 7, kDevLo = 0x100;  // where it lives on the "device"

int failures = 0;

uint32_t flatW0(unsigned op, bool glc) { return (0x37u << 26) | (op << 18) | (glc ? 1u << 16 : 0); }
uint32_t flatW1(unsigned vaddr, unsigned vdata, unsigned vdst) { return vaddr | vdata << 8 | vdst << 24; }
void flat(std::vector<uint32_t>& c, unsigned op, bool glc, unsigned vaddr, unsigned vdata, unsigned vdst) {
    c.push_back(flatW0(op, glc));
    c.push_back(flatW1(vaddr, vdata, vdst));
}
// v_mov_b32 v<d>, s<s>
uint32_t movS(unsigned d, unsigned s) { return 0x7e000200u | d << 17 | s; }

std::vector<uint32_t> prelude() {
    std::vector<uint32_t> c;
    c.push_back(movS(2, 0));
    c.push_back(movS(3, 1));
    c.push_back(movS(4, 2));
    c.push_back(movS(5, 3));
    for (unsigned k = 0; k < 4; ++k) c.push_back(movS(6 + k, 4 + k));
    return c;
}

struct Case {
    const char* name;
    std::vector<uint32_t> code;     // after the prelude
    uint32_t s[8];                  // user SGPRs
    std::vector<uint32_t> memory;   // region contents (dwords from guest base)
    std::vector<uint32_t> expect;   // expected region contents
};

bool runCase(const Case& t) {
    std::vector<uint32_t> code = prelude();
    code.insert(code.end(), t.code.begin(), t.code.end());
    code.push_back(0xbf810000u);  // s_endpgm
    vpgcn::LiftInput in;
    in.stage = vpgcn::ShaderStage::Compute;
    in.code = code;
    in.userSgprs = 8;
    vpgcn::LiftResult r = vpgcn::lift(in);
    for (auto& e : r.errors) std::printf("%s: lift error %s\n", t.name, e.c_str());
    if (!r.errors.empty()) return false;
    vpireval::Invocation inv;
    vpireval::Value tid;
    tid.type = vpir::kU32;
    for (auto& [b, v] : std::map<vpir::Builtin, vpireval::Value>{}) inv.builtins[b] = v;
    if (r.userDataSlot) {
        std::vector<uint8_t> ud(64);
        std::memcpy(ud.data(), t.s, sizeof t.s);
        inv.memory[*r.userDataSlot] = ud;
    }
    bool haveMap = false;
    for (const vpgcn::Binding& b : r.bindings) {
        if (b.source.kind != vpgcn::DescriptorSource::GuestMap) continue;
        haveMap = true;
        std::vector<uint8_t> map(vpgcn::kGuestMapBytes);
        uint32_t entry[8] = {uint32_t(kGuestBase), uint32_t(kGuestBase >> 32), kGuestSize, 0, kDevLo, kDevHi, 0, 0};
        uint32_t count = 1;
        std::memcpy(map.data(), &count, 4);
        std::memcpy(map.data() + 16, entry, sizeof entry);
        inv.memory[b.slot] = map;
    }
    if (!haveMap) {
        std::printf("%s: no GuestMap binding\n", t.name);
        return false;
    }
    std::vector<uint8_t> mem(kDevLo + kGuestSize, 0);
    std::memcpy(mem.data() + kDevLo, t.memory.data(), t.memory.size() * 4);
    inv.memory[0x10000 + kDevHi] = mem;
    std::string err;
    if (!vpireval::run(r.module, inv, err)) {
        std::printf("%s: eval error %s\n%s\n", t.name, err.c_str(), vpir::print(r.module).c_str());
        return false;
    }
    const std::vector<uint8_t>& got = inv.memory[0x10000 + kDevHi];
    bool ok = true;
    for (size_t k = 0; k < t.expect.size(); ++k) {
        uint32_t w = 0;
        if (kDevLo + 4 * k + 4 <= got.size()) std::memcpy(&w, &got[kDevLo + 4 * k], 4);
        if (w != t.expect[k]) {
            std::printf("%s: dword %zu got 0x%08x want 0x%08x\n", t.name, k, w, t.expect[k]);
            ok = false;
        }
    }
    return ok;
}

const uint32_t SRC_LO = uint32_t(kGuestBase), SRC_HI = uint32_t(kGuestBase >> 32);
// source at dword 0, destination at dword 16 (byte 64)
const uint32_t DST = 64;

Case mk(const char* name, std::vector<uint32_t> code, std::vector<uint32_t> memory, std::vector<uint32_t> expect,
        std::array<uint32_t, 4> data = {}) {
    Case c{name, std::move(code), {SRC_LO, SRC_HI, SRC_LO + DST, SRC_HI, data[0], data[1], data[2], data[3]}, std::move(memory),
           std::move(expect)};
    return c;
}

}  // namespace

int main() {
    std::vector<Case> cases;
    auto add = [&](Case c) { cases.push_back(std::move(c)); };
    auto loadStore = [&](unsigned lop, unsigned sop, unsigned n) {
        std::vector<uint32_t> c;
        flat(c, lop, false, 2, 0, 10);   // v10.. <- [v2:3]
        flat(c, sop, false, 4, 10, 0);   // [v4:5] <- v10..
        (void)n;
        return c;
    };
    std::vector<uint32_t> m(32, 0);
    m[0] = 0x11223344;
    m[1] = 0x8899aabb;
    m[2] = 0xccddeeff;
    m[3] = 0x01020304;
    auto dst = [&](std::vector<uint32_t> d) {
        std::vector<uint32_t> e(32, 0);
        e[0] = m[0], e[1] = m[1], e[2] = m[2], e[3] = m[3];
        for (size_t k = 0; k < d.size(); ++k) e[16 + k] = d[k];
        return e;
    };
    add(mk("load/store dword", loadStore(12, 28, 1), m, dst({0x11223344})));
    add(mk("load/store dwordx2", loadStore(13, 29, 2), m, dst({0x11223344, 0x8899aabb})));
    add(mk("load/store dwordx3", loadStore(15, 31, 3), m, dst({0x11223344, 0x8899aabb, 0xccddeeff})));
    add(mk("load/store dwordx4", loadStore(14, 30, 4), m, dst({0x11223344, 0x8899aabb, 0xccddeeff, 0x01020304})));
    {
        // sub-dword loads: address + 5 -> byte 1 of dword 1 (0xaa) / address + 6 -> 0x99 (sbyte -> 0xffffff99)
        std::vector<uint32_t> c;
        // v_add_i32 v2, vcc, 5, v2 (VOP2 op 0x25): src0 = inline constant 5 (code 133), vsrc1 = v2
        c.push_back(0x4a000000u | 2u << 17 | 2u << 9 | 133u);   // v_add_i32 v2, vcc, 5, v2
        flat(c, 8, false, 2, 0, 10);   // ubyte
        flat(c, 28, false, 4, 10, 0);
        add(mk("load ubyte", c, m, dst({0xaa})));
    }
    {
        std::vector<uint32_t> c;
        c.push_back(0x4a000000u | 2u << 17 | 2u << 9 | 134u);   // v2 += 6
        flat(c, 9, false, 2, 0, 10);   // sbyte
        flat(c, 28, false, 4, 10, 0);
        add(mk("load sbyte", c, m, dst({0xffffff99})));
        std::vector<uint32_t> d;
        d.push_back(0x4a000000u | 2u << 17 | 2u << 9 | 134u);
        flat(d, 10, false, 2, 0, 10);  // ushort at +6: 0x8899
        flat(d, 28, false, 4, 10, 0);
        add(mk("load ushort", d, m, dst({0x8899})));
        std::vector<uint32_t> e;
        e.push_back(0x4a000000u | 2u << 17 | 2u << 9 | 134u);
        flat(e, 11, false, 2, 0, 10);  // sshort at +6
        flat(e, 28, false, 4, 10, 0);
        add(mk("load sshort", e, m, dst({0xffff8899})));
    }
    {
        // byte / short stores merge into the existing dword
        std::vector<uint32_t> c;
        c.push_back(0x4a000000u | 4u << 17 | 4u << 9 | 129u);   // v4 += 1 (destination + 1)
        flat(c, 24, false, 4, 6, 0);                            // store_byte [v4:5] <- v6 (0xab)
        std::vector<uint32_t> e(32, 0);
        e[16] = 0x0000ab00u;
        std::vector<uint32_t> init(32, 0);
        add(mk("store byte", c, init, e, {0xab, 0, 0, 0}));
        std::vector<uint32_t> d;
        d.push_back(0x4a000000u | 4u << 17 | 4u << 9 | 130u);   // v4 += 2
        flat(d, 26, false, 4, 6, 0);
        std::vector<uint32_t> e2(32, 0);
        e2[16] = 0xbeef0000u;
        add(mk("store short", d, init, e2, {0xdeadbeef, 0, 0, 0}));
    }
    auto atomic = [&](unsigned op, bool glc, unsigned vdata) {
        std::vector<uint32_t> c;
        flat(c, op, glc, 2, vdata, 10);        // v10.. <- old
        if (glc) flat(c, op >= 80 ? 29 : 28, false, 4, 10, 0);  // return value to the destination
        return c;
    };
    std::vector<uint32_t> a(32, 0);
    a[0] = 5;
    auto ex = [&](uint32_t v0, uint32_t ret) {
        std::vector<uint32_t> e(32, 0);
        e[0] = v0;
        e[16] = ret;
        return e;
    };
    add(mk("atomic add", atomic(50, true, 6), a, ex(8, 5), {3, 0, 0, 0}));
    add(mk("atomic sub", atomic(51, true, 6), a, ex(2, 5), {3, 0, 0, 0}));
    add(mk("atomic swap", atomic(48, true, 6), a, ex(9, 5), {9, 0, 0, 0}));
    add(mk("atomic umin", atomic(54, true, 6), a, ex(3, 5), {3, 0, 0, 0}));
    add(mk("atomic umax", atomic(56, true, 6), a, ex(7, 5), {7, 0, 0, 0}));
    add(mk("atomic smin", atomic(53, true, 6), a, ex(0xfffffffe, 5), {0xfffffffe, 0, 0, 0}));
    add(mk("atomic smax", atomic(55, true, 6), a, ex(5, 5), {0xfffffffe, 0, 0, 0}));
    add(mk("atomic and", atomic(57, true, 6), a, ex(4, 5), {6, 0, 0, 0}));
    add(mk("atomic or", atomic(58, true, 6), a, ex(7, 5), {2, 0, 0, 0}));
    add(mk("atomic xor", atomic(59, true, 6), a, ex(6, 5), {3, 0, 0, 0}));
    add(mk("atomic inc wraps", atomic(60, true, 6), a, ex(0, 5), {5, 0, 0, 0}));
    add(mk("atomic inc", atomic(60, true, 6), a, ex(6, 5), {9, 0, 0, 0}));
    add(mk("atomic dec", atomic(61, true, 6), a, ex(4, 5), {9, 0, 0, 0}));
    add(mk("atomic dec resets", atomic(61, true, 6), a, ex(3, 5), {3, 0, 0, 0}));
    add(mk("atomic cmpswap hit", atomic(49, true, 6), a, ex(9, 5), {9, 5, 0, 0}));
    add(mk("atomic cmpswap miss", atomic(49, true, 6), a, ex(5, 5), {9, 4, 0, 0}));
    add(mk("atomic without return", atomic(50, false, 6), a, ex(6, 0), {1, 0, 0, 0}));
    {
        // 64-bit add with carry: 0x00000000ffffffff + 1
        std::vector<uint32_t> w(32, 0);
        w[0] = 0xffffffff;
        std::vector<uint32_t> e(32, 0);
        e[0] = 0;
        e[1] = 1;
        e[16] = 0xffffffff;
        e[17] = 0;
        add(mk("atomic add_x2 carry", atomic(82, true, 6), w, e, {1, 0, 0, 0}));
        std::vector<uint32_t> e2(32, 0);
        e2[0] = 0xffffffff;
        e2[1] = 0;
        e2[16] = 0;
        e2[17] = 1;
        std::vector<uint32_t> w2(32, 0);
        w2[1] = 1;
        add(mk("atomic sub_x2 borrow", atomic(83, true, 6), w2, e2, {1, 0, 0, 0}));
    }
    {
        // a pointer outside the region reads zeros; its stores are dropped
        std::vector<uint32_t> c;
        c.push_back(movS(2, 0));  // keep the prelude's v2; the source address is s0 = outside below
        flat(c, 12, false, 2, 0, 10);
        flat(c, 28, false, 4, 10, 0);
        Case t = mk("load outside the map", c, m, dst({0}));
        t.s[1] = 5;  // hi = 5: no such region
        t.memory = m;
        std::vector<uint32_t> e(32, 0);
        e[0] = m[0], e[1] = m[1], e[2] = m[2], e[3] = m[3];
        e[16] = 0;
        t.expect = e;
        add(std::move(t));
        std::vector<uint32_t> d;
        flat(d, 28, false, 2, 6, 0);  // store to the outside address
        Case u = mk("store outside the map", d, m, std::vector<uint32_t>(m.begin(), m.end()), {0x1234, 0, 0, 0});
        u.s[1] = 5;
        add(std::move(u));
        // a dwordx4 access that runs past the end of the region is dropped as a whole
        std::vector<uint32_t> c2;
        flat(c2, 30, false, 2, 6, 0);
        Case v = mk("store crossing the end", c2, m, std::vector<uint32_t>(m.begin(), m.end()), {1, 2, 3, 4});
        v.s[0] = SRC_LO + kGuestSize - 8;
        add(std::move(v));
    }
    for (const Case& t : cases) {
        bool ok = runCase(t);
        if (!ok) ++failures;
    }
    std::printf("%zu FLAT cases, %d failed\n", cases.size(), failures);
    return failures ? 1 : 0;
}
