// FLAT instructions of the GCN lifter: included inside `struct Lifter` (lift.cpp).
//
// FLAT addresses memory with a raw 64-bit pointer in a VGPR pair (vaddr). On the console every such
// pointer is a guest GPU virtual address, which means nothing to the host GPU. The lowering goes
// through the `GuestMap` binding (DescriptorSource::GuestMap): a small constant buffer the renderer
// fills at draw time with the guest regions it has mapped as GPU buffers,
//
//     u32 count, pad[3]                                             (16 bytes)
//     count x { baseLo, baseHi, sizeLo, sizeHi, devLo, devHi, 0, 0 } (32 bytes each, sorted by base,
//                                                                     at most kGuestMapEntries)
//
// where dev is the buffer's device address (Device::bufferAddress). The shader finds the last region
// whose base is <= the pointer with an unrolled binary search (8 steps), checks that the access lies
// inside it and forms `dev + (va - base)`, a u32x2 operand of the VPIR `device` memory target (the
// VK_KHR_buffer_device_address / Metal gpuAddress path). A pointer outside every region reads zeros and
// drops writes and atomics (the hardware would fault). Accesses through it are not bounds-checked by the
// backend, which is why the check is made here.
//
// Not modelled: the FLAT aperture split (flat_scratch / LDS / private apertures): every pointer is
// global. 64-bit atomics have no 64-bit atomic in VPIR: add / sub (carry through the low dword), and,
// or, xor and swap act per dword, atomically for each; the others (min, max, inc, dec, cmpswap) are
// a non-atomic read-modify-write and are noted.

struct FlatAddr {
    ValueId ok;      // the access lies inside a mapped region
    ValueId lo, hi;  // the device address of the first byte
};

uint32_t guestMapSlot() {
    DescriptorSource s;
    s.kind = DescriptorSource::GuestMap;
    return resource(V::ResourceKind::ConstantBuffer, s, 0, V::Access::Read, "guest");
}

// a < b for 64-bit values given as (hi, lo) dwords
ValueId lt64(ValueId ahi, ValueId alo, ValueId bhi, ValueId blo) {
    auto B = V::kBool;
    return bin(V::Op::Or, bin(V::Op::Lt, ahi, bhi, B), bin(V::Op::And, bin(V::Op::Eq, ahi, bhi, B), bin(V::Op::Lt, alo, blo, B), B), B);
}

FlatAddr flatTranslate(ValueId vlo, ValueId vhi, uint32_t bytes) {
    auto U = V::kU32;
    auto B = V::kBool;
    const uint32_t slot = guestMapSlot();
    ValueId count = b->load(slot, u(0), U);
    auto entry = [&](ValueId index, unsigned word) {
        ValueId at = bin(V::Op::Add, bin(V::Op::Mul, index, u(32), U), u(16 + 4 * word), U);
        return b->load(slot, at, U);
    };
    ValueId pos = u(0);
    for (unsigned step = kGuestMapEntries / 2; step; step >>= 1) {
        ValueId cand = bin(V::Op::Add, pos, u(step), U);
        ValueId blo = entry(cand, 0), bhi = entry(cand, 1);
        ValueId take = bin(V::Op::And, bin(V::Op::Lt, cand, count, B), notB(lt64(vhi, vlo, bhi, blo)), B);
        pos = sel(take, cand, pos, U);
    }
    ValueId blo = entry(pos, 0), bhi = entry(pos, 1), slo = entry(pos, 2), shi = entry(pos, 3);
    ValueId dlo = entry(pos, 4), dhi = entry(pos, 5);
    ValueId inside = bin(V::Op::And, bin(V::Op::Gt, count, u(0), B), notB(lt64(vhi, vlo, bhi, blo)), B);
    // off = va - base; end = off + bytes must not pass the region's size
    ValueId olo = bin(V::Op::Sub, vlo, blo, U);
    ValueId ohi = bin(V::Op::Sub, bin(V::Op::Sub, vhi, bhi, U), b2u(bin(V::Op::Lt, vlo, blo, B)), U);
    ValueId elo = bin(V::Op::Add, olo, u(bytes), U);
    ValueId ehi = bin(V::Op::Add, ohi, b2u(bin(V::Op::Lt, elo, olo, B)), U);
    ValueId fits = notB(lt64(shi, slo, ehi, elo));  // end <= size
    FlatAddr r;
    r.ok = bin(V::Op::And, inside, fits, B);
    r.lo = bin(V::Op::Add, dlo, olo, U);
    r.hi = bin(V::Op::Add, bin(V::Op::Add, dhi, ohi, U), b2u(bin(V::Op::Lt, r.lo, dlo, B)), U);
    return r;
}

// The device address `k` bytes past a translated one (as the u32x2 operand of the `device` target).
ValueId flatAt(const FlatAddr& a, uint32_t k) {
    auto U = V::kU32;
    ValueId lo = k ? bin(V::Op::Add, a.lo, u(k), U) : a.lo;
    ValueId hi = k ? bin(V::Op::Add, a.hi, b2u(bin(V::Op::Lt, lo, a.lo, V::kBool)), U) : a.hi;
    ValueId parts[] = {lo, hi};
    return b->construct(parts, V::vec(U, 2));
}

void liftFlat(size_t idx) {
    const Instruction& x = prog[idx].in;
    std::string_view name = mnemonic(x.op);
    auto U = V::kU32;
    auto B = V::kBool;
    const bool atomic = name.starts_with("flat_atomic");
    const bool store = name.starts_with("flat_store");
    const bool wide = name.ends_with("_x2");
    const unsigned data = x.src[1].kind == OperandKind::Vgpr ? x.src[1].index : 0;
    if (x.src[0].kind != OperandKind::Vgpr || x.src[0].count != 2) return error("FLAT address is not a VGPR pair");
    ValueId vlo = b->loadVar(vVar(x.src[0].index)), vhi = b->loadVar(vVar(x.src[0].index + 1));
    auto guarded = [&](auto&& body) {
        if (writesExec) b->beginIf(b->loadVar(execV()));
        body();
        if (writesExec) b->end();
    };
    auto reg = [&](unsigned k) { return b->loadVar(vVar(data + k)); };

    if (atomic) {
        const bool glc = x.mem.glc;
        uint32_t dwords = wide ? 2 : 1;
        // fcmpswap / cmpswap take {swap value, compare value}: twice the data of the others
        const bool cmp = name.find("cmpswap") != std::string_view::npos;
        std::string_view op = name.substr(12);  // after "flat_atomic_"
        if (wide) op = op.substr(0, op.size() - 3);
        std::vector<uint32_t> ret;
        for (unsigned k = 0; k < dwords; ++k) ret.push_back(b->variable(U));
        for (uint32_t r : ret) b->storeVar(r, u(0));
        auto body = [&] {
            FlatAddr a = flatTranslate(vlo, vhi, 4 * dwords);
            b->beginIf(a.ok);
            auto atom = [&](V::Op o, unsigned k, ValueId v, V::Type t = V::kU32) {
                ValueId vals[] = {t == V::kI32 ? asI(v) : v};
                ValueId old = b->atomic(o, V::kTargetDevice, flatAt(a, 4 * k), vals, t);
                return t == V::kI32 ? bit(old, U) : old;
            };
            // a compare-exchange loop applying f(old) -> new on dword k
            auto casLoop = [&](unsigned k, auto&& f) {
                b->beginLoop();
                ValueId old = b->load(V::kTargetDevice, flatAt(a, 4 * k), U);
                ValueId nv = f(old);
                ValueId vals[] = {old, nv};
                ValueId prev = b->atomic(V::Op::AtomicCompareExchange, V::kTargetDevice, flatAt(a, 4 * k), vals, U);
                b->storeVar(ret[k], old);
                b->beginIf(bin(V::Op::Eq, prev, old, B));
                b->breakLoop();
                b->end();
                b->end();
            };
            if (op == "swap" || op == "add" || op == "sub" || op == "and" || op == "or" || op == "xor" || op == "smin" ||
                op == "umin" || op == "smax" || op == "umax") {
                V::Op o = op == "swap" ? V::Op::AtomicExchange : op == "add" ? V::Op::AtomicAdd : op == "sub" ? V::Op::AtomicSub
                        : op == "and" ? V::Op::AtomicAnd : op == "or" ? V::Op::AtomicOr : op == "xor" ? V::Op::AtomicXor
                        : op.ends_with("min") ? V::Op::AtomicMin : V::Op::AtomicMax;
                const bool sign = op[0] == 's' && op != "swap" && op != "sub";
                if (!wide) {
                    b->storeVar(ret[0], atom(o, 0, reg(0), sign ? V::kI32 : U));
                } else if (o == V::Op::AtomicMin || o == V::Op::AtomicMax) {
                    // non-atomic 64-bit read-modify-write
                    ValueId ol = b->load(V::kTargetDevice, flatAt(a, 0), U), oh = b->load(V::kTargetDevice, flatAt(a, 4), U);
                    ValueId lessHi = sign ? bin(V::Op::Lt, asI(reg(1)), asI(oh), B) : bin(V::Op::Lt, reg(1), oh, B);
                    ValueId lessLo = bin(V::Op::Lt, reg(0), ol, B);
                    ValueId less = bin(V::Op::Or, lessHi, bin(V::Op::And, bin(V::Op::Eq, reg(1), oh, B), lessLo, B), B);
                    ValueId takeNew = o == V::Op::AtomicMin ? less : notB(bin(V::Op::Or, less, bin(V::Op::And, bin(V::Op::Eq, reg(1), oh, B), bin(V::Op::Eq, reg(0), ol, B), B), B));
                    b->store(V::kTargetDevice, flatAt(a, 0), sel(takeNew, reg(0), ol, U));
                    b->store(V::kTargetDevice, flatAt(a, 4), sel(takeNew, reg(1), oh, U));
                    b->storeVar(ret[0], ol);
                    b->storeVar(ret[1], oh);
                } else if (o == V::Op::AtomicAdd || o == V::Op::AtomicSub) {
                    ValueId ol = atom(o, 0, reg(0));
                    ValueId carry = o == V::Op::AtomicAdd ? b2u(bin(V::Op::Lt, bin(V::Op::Add, ol, reg(0), U), ol, B))
                                                          : b2u(bin(V::Op::Lt, ol, reg(0), B));
                    ValueId hiOperand = bin(o == V::Op::AtomicAdd ? V::Op::Add : V::Op::Add, reg(1), carry, U);
                    b->storeVar(ret[0], ol);
                    b->storeVar(ret[1], atom(o, 1, hiOperand));
                } else {  // swap, and, or, xor: bitwise per dword
                    b->storeVar(ret[0], atom(o, 0, reg(0)));
                    b->storeVar(ret[1], atom(o, 1, reg(1)));
                }
            } else if (op == "cmpswap") {
                if (!wide) {
                    // vdata = {swap value, compare value}
                    ValueId vals[] = {reg(1), reg(0)};
                    b->storeVar(ret[0], b->atomic(V::Op::AtomicCompareExchange, V::kTargetDevice, flatAt(a, 0), vals, U));
                } else {  // non-atomic: vdata = {swap lo, swap hi, compare lo, compare hi}
                    ValueId ol = b->load(V::kTargetDevice, flatAt(a, 0), U), oh = b->load(V::kTargetDevice, flatAt(a, 4), U);
                    ValueId same = bin(V::Op::And, bin(V::Op::Eq, ol, reg(2), B), bin(V::Op::Eq, oh, reg(3), B), B);
                    b->store(V::kTargetDevice, flatAt(a, 0), sel(same, reg(0), ol, U));
                    b->store(V::kTargetDevice, flatAt(a, 4), sel(same, reg(1), oh, U));
                    b->storeVar(ret[0], ol);
                    b->storeVar(ret[1], oh);
                }
            } else if (op == "fcmpswap" && !wide) {
                ValueId cmpv = reg(1);
                casLoop(0, [&](ValueId old) { return sel(bin(V::Op::Eq, asF(old), asF(cmpv), B), reg(0), old, U); });
            } else if (op == "inc" && !wide) {
                casLoop(0, [&](ValueId old) { return sel(notB(bin(V::Op::Lt, old, reg(0), B)), u(0), bin(V::Op::Add, old, u(1), U), U); });
            } else if (op == "dec" && !wide) {
                casLoop(0, [&](ValueId old) {
                    ValueId reset = bin(V::Op::Or, bin(V::Op::Eq, old, u(0), B), bin(V::Op::Gt, old, reg(0), B), B);
                    return sel(reset, reg(0), bin(V::Op::Sub, old, u(1), U), U);
                });
            } else if ((op == "fmin" || op == "fmax") && !wide) {
                const bool isMin = op == "fmin";
                casLoop(0, [&](ValueId old) {
                    ValueId o = asF(old), v = asF(reg(0));
                    return bit(b->binary(isMin ? V::Op::Min : V::Op::Max, o, v, V::kF32), U);
                });
            } else if (wide) {
                // 64-bit inc / dec / fcmpswap / fmin / fmax: non-atomic read-modify-write of the pair
                ValueId ol = b->load(V::kTargetDevice, flatAt(a, 0), U), oh = b->load(V::kTargetDevice, flatAt(a, 4), U);
                ValueId nlo, nhi;
                if (op == "inc" || op == "dec") {
                    ValueId s0 = reg(0), s1 = reg(1);
                    ValueId one = op == "inc" ? bin(V::Op::Add, ol, u(1), U) : bin(V::Op::Sub, ol, u(1), U);
                    ValueId carry = op == "inc" ? b2u(bin(V::Op::Lt, one, ol, B)) : b2u(bin(V::Op::Gt, one, ol, B));
                    ValueId stepHi = op == "inc" ? bin(V::Op::Add, oh, carry, U) : bin(V::Op::Sub, oh, carry, U);
                    ValueId reset;
                    if (op == "inc") reset = notB(lt64(oh, ol, s1, s0));  // old >= src
                    else reset = bin(V::Op::Or, bin(V::Op::And, bin(V::Op::Eq, oh, u(0), B), bin(V::Op::Eq, ol, u(0), B), B), lt64(s1, s0, oh, ol), B);
                    nlo = sel(reset, op == "inc" ? u(0) : s0, one, U);
                    nhi = sel(reset, op == "inc" ? u(0) : s1, stepHi, U);
                } else {
                    ValueId old = f64Pair(ol, oh);
                    ValueId r;
                    if (op == "fcmpswap") {
                        ValueId same = f64Call(V::F64Fn::Cmp, {old, f64Pair(reg(2), reg(3)), u(2), f64Mode()});
                        nlo = sel(same, reg(0), ol, U);
                        nhi = sel(same, reg(1), oh, U);
                    } else {
                        r = f64Call(op == "fmin" ? V::F64Fn::Min : V::F64Fn::Max, {old, f64Pair(reg(0), reg(1)), f64Mode()});
                        nlo = f64Lo(r);
                        nhi = f64Hi(r);
                    }
                }
                b->store(V::kTargetDevice, flatAt(a, 0), nlo);
                b->store(V::kTargetDevice, flatAt(a, 4), nhi);
                b->storeVar(ret[0], ol);
                b->storeVar(ret[1], oh);
            } else {
                error("unsupported FLAT atomic " + std::string(name));
            }
            if (wide && op != "swap" && op != "and" && op != "or" && op != "xor" && op != "add" && op != "sub" && op != "cmpswap")
                note("64-bit FLAT atomic " + std::string(op) + " is not atomic");
            else if (wide && (op == "add" || op == "sub" || op == "swap"))
                note("64-bit FLAT atomic " + std::string(op) + " is atomic per dword only");
            b->end();
        };
        (void)cmp;
        guarded(body);
        if (glc && x.dst.kind == OperandKind::Vgpr)
            for (unsigned k = 0; k < dwords; ++k) writeV(x.dst.index + k, b->loadVar(ret[k]));
        return;
    }

    if (store) {
        unsigned dw = name.ends_with("x4") ? 4 : name.ends_with("x3") ? 3 : name.ends_with("x2") ? 2 : 1;
        const bool sub = name.ends_with("_byte") || name.ends_with("_short");
        auto body = [&] {
            if (sub) {
                const unsigned width = name.ends_with("_byte") ? 8 : 16;
                ValueId word = bin(V::Op::And, vlo, u(~3u), U);
                FlatAddr a = flatTranslate(word, vhi, 4);
                b->beginIf(a.ok);
                ValueId sh = bin(V::Op::Shl, bin(V::Op::And, vlo, u(3), U), u(3), U);
                uint32_t low = width == 8 ? 0xffu : 0xffffu;
                ValueId mask = bin(V::Op::Shl, u(low), sh, U);
                ValueId field = bin(V::Op::Shl, bin(V::Op::And, reg(0), u(low), U), sh, U);
                ValueId keep[] = {b->unary(V::Op::Not, mask, U)};
                b->atomic(V::Op::AtomicAnd, V::kTargetDevice, flatAt(a, 0), keep, U);
                ValueId set[] = {field};
                b->atomic(V::Op::AtomicOr, V::kTargetDevice, flatAt(a, 0), set, U);
                b->end();
                return;
            }
            FlatAddr a = flatTranslate(vlo, vhi, 4 * dw);
            b->beginIf(a.ok);
            for (unsigned k = 0; k < dw; ++k) b->store(V::kTargetDevice, flatAt(a, 4 * k), reg(k));
            b->end();
        };
        guarded(body);
        return;
    }

    // Loads
    const bool sub = name.ends_with("_ubyte") || name.ends_with("_sbyte") || name.ends_with("_ushort") || name.ends_with("_sshort");
    unsigned dw = sub ? 1 : name.ends_with("x4") ? 4 : name.ends_with("x3") ? 3 : name.ends_with("x2") ? 2 : 1;
    std::vector<uint32_t> tmp;
    for (unsigned k = 0; k < dw; ++k) {
        tmp.push_back(b->variable(U));
        b->storeVar(tmp[k], u(0));
    }
    ValueId addrLo = sub ? bin(V::Op::And, vlo, u(~3u), U) : vlo;
    FlatAddr a = flatTranslate(addrLo, vhi, 4 * dw);
    b->beginIf(a.ok);
    for (unsigned k = 0; k < dw; ++k) b->storeVar(tmp[k], b->load(V::kTargetDevice, flatAt(a, 4 * k), U));
    b->end();
    if (sub) {
        ValueId sh = bin(V::Op::Shl, bin(V::Op::And, vlo, u(3), U), u(3), U);
        unsigned width = name.ends_with("byte") ? 8 : 16;
        ValueId field = bin(V::Op::Shr, b->loadVar(tmp[0]), sh, U);
        writeV(x.dst.index, bfe(field, u(0), u(width), name.ends_with("_sbyte") || name.ends_with("_sshort")));
    } else {
        for (unsigned k = 0; k < dw; ++k) writeV(x.dst.index + k, b->loadVar(tmp[k]));
    }
    if (x.mem.tfe) writeV(x.dst.index + dw, u(0));
}
