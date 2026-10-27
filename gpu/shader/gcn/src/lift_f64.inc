// 64-bit float (f64) vector instructions of the GCN lifter: included inside `struct Lifter`
// (lift.cpp). A double is a pair of VGPRs (low first) and a vec2<u32> in VPIR; the arithmetic is the
// soft-float library behind the VPIR `f64` op (vpir/f64.h), parameterised by the MODE register: rounding
// mode (FP_ROUND double), denormal handling (FP_DENORM double), IEEE and DX10_CLAMP bits.
//
// 64-bit sources: float inline constants are doubles (0.5, 1.0, ... and 1/(2pi)); integer inline
// constants sign-extend to 64 bits; a literal is the HIGH dword (the low one is 0). VOP3 abs / neg act on the
// sign bit of the high dword; omod scales the result by 2^+-1 / 2^2; clamp saturates to [0, 1].
//
// Not modelled: exception flags, and the hardware's approximate rcp / rsq / sqrt estimates (results are
// correctly rounded; rsq within 0.502 ulp), see docs/isa-f64.md.

// The MODE register as the f64 functions take it.
ValueId f64Mode() { return modeDynamic ? b->loadVar(modeV()) : u(curMode); }

ValueId f64Call(V::F64Fn fn, std::initializer_list<ValueId> args) {
    V::Inst in;
    in.op = V::Op::F64;
    in.type = V::f64CodeType(V::f64FnInfo(fn).result);
    in.args.assign(args.begin(), args.end());
    in.lits = {uint32_t(fn)};
    return b->emit(in);
}
ValueId f64Pair(ValueId lo, ValueId hi) {
    ValueId parts[] = {lo, hi};
    return b->construct(parts, V::vec(V::kU32, 2));
}
ValueId f64Lo(ValueId d) { return b->extract(d, 0, V::kU32); }
ValueId f64Hi(ValueId d) { return b->extract(d, 1, V::kU32); }

// A 64-bit float source operand with VOP3 modifiers, as vec2<u32>.
ValueId srcD(const Instruction& x, unsigned i) {
    const Operand& o = x.src[i];
    ValueId lo, hi;
    if (o.kind == OperandKind::Const) {
        if (o.isFloat) {
            uint64_t d = o.code == 248 ? 0x3fc45f306dc9c882ull : V::f64::fromF32(o.value);  // 1/(2pi) as the hardware (and LLVM) take it
            lo = u(uint32_t(d));
            hi = u(uint32_t(d >> 32));
        } else {
            lo = u(o.value);
            hi = u(int32_t(o.value) < 0 ? 0xffffffffu : 0u);
        }
    } else if (o.kind == OperandKind::Literal) {
        lo = u(0);
        hi = u(o.value);
    } else {
        lo = readS(o, 0);
        hi = readS(o, 1);
    }
    if (x.abs >> i & 1) hi = bin(V::Op::And, hi, u(0x7fffffffu), V::kU32);
    if (x.neg >> i & 1) hi = bin(V::Op::Xor, hi, u(0x80000000u), V::kU32);
    return f64Pair(lo, hi);
}

void writeD(const Instruction& x, ValueId d) {
    writeV(x.dst.index, f64Lo(d));
    writeV(x.dst.index + 1, f64Hi(d));
}
// omod and clamp, then the pair into the destination VGPRs.
void resultD(const Instruction& x, ValueId d) {
    if (x.omod) d = f64Call(V::F64Fn::Ldexp, {d, u(x.omod == 1 ? 1u : x.omod == 2 ? 2u : 0xffffffffu), f64Mode()});
    if (x.clamp) d = f64Call(V::F64Fn::Sat, {d, f64Mode()});
    writeD(x, d);
}

// v_cmp[x|s|sx]_<cond>_f64 and the class test.
ValueId cmpF64(const Instruction& x) {
    std::string_view n = mnemonic(x.op);
    size_t p = n.find('_', 2);  // end of v_cmp / v_cmpx / v_cmps / v_cmpsx
    std::string_view cond = n.substr(p + 1, n.rfind('_') - p - 1);
    static const char* const kConds[16] = {"f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
    for (unsigned i = 0; i < 16; ++i)
        if (cond == kConds[i]) return f64Call(V::F64Fn::Cmp, {srcD(x, 0), srcD(x, 1), u(i), f64Mode()});
    error("unknown 64-bit float compare " + std::string(n));
    return b->constBool(false);
}
ValueId classF64(const Instruction& x) { return f64Call(V::F64Fn::Class, {srcD(x, 0), srcU(x, 1), f64Mode()}); }

// The f64 vector ALU ops (not the compares, which share liftVop's mask handling). False: not an f64 op.
bool liftF64(const Instruction& x) {
    using F = V::F64Fn;
    G op = x.op;
    ValueId md = f64Mode();
    auto un = [&](F fn) { resultD(x, f64Call(fn, {srcD(x, 0), md})); };
    auto bi = [&](F fn) { resultD(x, f64Call(fn, {srcD(x, 0), srcD(x, 1), md})); };
    switch (op) {
        case G::v_add_f64: bi(F::Add); return true;
        case G::v_mul_f64: bi(F::Mul); return true;
        case G::v_min_f64: bi(F::Min); return true;
        case G::v_max_f64: bi(F::Max); return true;
        case G::v_fma_f64: resultD(x, f64Call(F::Fma, {srcD(x, 0), srcD(x, 1), srcD(x, 2), md})); return true;
        case G::v_ldexp_f64: resultD(x, f64Call(F::Ldexp, {srcD(x, 0), srcU(x, 1), md})); return true;
        case G::v_trunc_f64: un(F::Trunc); return true;
        case G::v_ceil_f64: un(F::Ceil); return true;
        case G::v_rndne_f64: un(F::Rndne); return true;
        case G::v_floor_f64: un(F::Floor); return true;
        case G::v_fract_f64: un(F::Fract); return true;
        case G::v_rcp_f64: un(F::Rcp); return true;
        case G::v_rcp_clamp_f64: un(F::RcpClamp); return true;
        case G::v_rsq_f64: un(F::Rsq); return true;
        case G::v_rsq_clamp_f64: un(F::RsqClamp); return true;
        case G::v_sqrt_f64: un(F::Sqrt); return true;
        case G::v_frexp_mant_f64: un(F::FrexpMant); return true;
        case G::v_frexp_exp_i32_f64: resultU(x, f64Call(F::FrexpExp, {srcD(x, 0), md})); return true;
        case G::v_cvt_i32_f64: resultU(x, f64Call(F::ToI32, {srcD(x, 0), md})); return true;
        case G::v_cvt_u32_f64: resultU(x, f64Call(F::ToU32, {srcD(x, 0), md})); return true;
        case G::v_cvt_f64_i32: resultD(x, f64Call(F::FromI32, {srcU(x, 0), md})); return true;
        case G::v_cvt_f64_u32: resultD(x, f64Call(F::FromU32, {srcU(x, 0), md})); return true;
        case G::v_cvt_f64_f32: resultD(x, f64Call(F::FromF32, {bit(srcF(x, 0), V::kU32), md})); return true;
        case G::v_cvt_f32_f64: {
            // (rounding by MODE's single-precision field, denormal results by its f32 output flag)
            ValueId r = f64Call(F::ToF32, {srcD(x, 0), md});
            resultF(x, asF(r));
            return true;
        }
        case G::v_div_fixup_f64: resultD(x, f64Call(F::DivFixup, {srcD(x, 0), srcD(x, 1), srcD(x, 2), md})); return true;
        case G::v_div_fmas_f64:
            resultD(x, f64Call(F::DivFmas, {srcD(x, 0), srcD(x, 1), srcD(x, 2), b2u(b->loadVar(vccV())), md}));
            return true;
        case G::v_div_scale_f64: {
            ValueId r = f64Call(F::DivScale, {srcD(x, 0), srcD(x, 1), srcD(x, 2), md});
            ValueId vcc = nz(b->extract(r, 2, V::kU32));
            writeMask(x.sdst, bin(V::Op::And, vcc, b->loadVar(execV()), V::kBool));
            resultD(x, f64Pair(b->extract(r, 0, V::kU32), b->extract(r, 1, V::kU32)));
            return true;
        }
        case G::v_trig_preop_f64: resultD(x, f64Call(F::TrigPreop, {srcD(x, 0), srcU(x, 1), md})); return true;
        default: return false;
    }
}
