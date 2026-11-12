#include "vpir/builder.h"

#include <bit>
#include <cassert>
#include <stdexcept>

namespace vpir {

FunctionBuilder::FunctionBuilder(Module& module, std::string name, Type returnType)
    : module_(module), index_(module.functions.size()) {
    Function f;
    f.name = std::move(name);
    f.returnType = returnType;
    module_.functions.push_back(std::move(f));
}

// Resolved on every access: holding references would dangle as blocks grow.
Function& FunctionBuilder::fn() { return module_.functions[index_]; }

Block& FunctionBuilder::current() { return open_.empty() ? fn().body : *open_.back().inner; }

ValueId FunctionBuilder::newValue(Type t) {
    fn().valueTypes.push_back(t);
    return ValueId(fn().valueTypes.size() - 1);
}

void FunctionBuilder::push(Node n) {
    if (limit_ && ++nodes_ > limit_) throw std::length_error("function exceeds " + std::to_string(limit_) + " nodes");
    current().push_back(std::move(n));
}

ValueId FunctionBuilder::param(Type t) {
    ValueId id = newValue(t);
    fn().params.push_back(id);
    return id;
}

uint32_t FunctionBuilder::variable(Type t) {
    fn().vars.push_back({t});
    return uint32_t(fn().vars.size() - 1);
}

ValueId FunctionBuilder::emit(Inst inst) {
    inst.result = inst.type.isVoid() ? kNoValue : newValue(inst.type);
    ValueId r = inst.result;
    Node n;
    n.inst = std::move(inst);
    push(std::move(n));
    return r;
}

namespace {
Inst make(Op op, Type type, uint32_t sym = 0) {
    Inst in;
    in.op = op;
    in.type = type;
    in.sym = sym;
    return in;
}
}  // namespace

ValueId FunctionBuilder::constant(Type t, std::initializer_list<uint32_t> bits) {
    Inst in = make(Op::Const, t);
    in.lits.assign(bits);
    return emit(std::move(in));
}
ValueId FunctionBuilder::constF32(float v) { return constant(kF32, {std::bit_cast<uint32_t>(v)}); }
ValueId FunctionBuilder::constU32(uint32_t v) { return constant(kU32, {v}); }
ValueId FunctionBuilder::constI32(int32_t v) { return constant(kI32, {uint32_t(v)}); }
ValueId FunctionBuilder::constBool(bool v) { return constant(kBool, {v ? 1u : 0u}); }

ValueId FunctionBuilder::nary(Op op, std::span<const ValueId> args, Type t, bool precise) {
    Inst in = make(op, t);
    in.precise = precise;
    in.args.assign(args.begin(), args.end());
    return emit(std::move(in));
}
ValueId FunctionBuilder::unary(Op op, ValueId a, Type t, bool precise) {
    const ValueId v[] = {a};
    return nary(op, v, t, precise);
}
ValueId FunctionBuilder::binary(Op op, ValueId a, ValueId b, Type t, bool precise) {
    const ValueId v[] = {a, b};
    return nary(op, v, t, precise);
}
ValueId FunctionBuilder::ternary(Op op, ValueId a, ValueId b, ValueId c, Type t, bool precise) {
    const ValueId v[] = {a, b, c};
    return nary(op, v, t, precise);
}

ValueId FunctionBuilder::loadVar(uint32_t var) { return emit(make(Op::LoadVar, fn().vars[var].type, var)); }
void FunctionBuilder::storeVar(uint32_t var, ValueId v) {
    Inst in = make(Op::StoreVar, kVoid, var);
    in.args = {v};
    emit(std::move(in));
}
ValueId FunctionBuilder::loadInput(uint32_t input, Type t) { return emit(make(Op::LoadInput, t, input)); }
void FunctionBuilder::storeOutput(uint32_t output, ValueId v) {
    Inst in = make(Op::StoreOutput, kVoid, output);
    in.args = {v};
    emit(std::move(in));
}
ValueId FunctionBuilder::call(uint32_t function, std::span<const ValueId> args, Type t) {
    Inst in = make(Op::Call, t, function);
    in.args.assign(args.begin(), args.end());
    return emit(std::move(in));
}

ValueId FunctionBuilder::extract(ValueId v, uint32_t index, Type t) {
    Inst in = make(Op::Extract, t);
    in.args = {v};
    in.lits = {index};
    return emit(std::move(in));
}
ValueId FunctionBuilder::construct(std::span<const ValueId> parts, Type t) { return nary(Op::Construct, parts, t); }

ValueId FunctionBuilder::load(uint32_t target, ValueId address, Type t) {
    Inst in = make(Op::Load, t, target);
    in.args = {address};
    return emit(std::move(in));
}
void FunctionBuilder::store(uint32_t target, ValueId address, ValueId v) {
    Inst in = make(Op::Store, kVoid, target);
    in.args = {address, v};
    emit(std::move(in));
}
ValueId FunctionBuilder::atomic(Op op, uint32_t target, ValueId address, std::span<const ValueId> values, Type t) {
    Inst in = make(op, t, target);
    in.args = {address};
    in.args.insert(in.args.end(), values.begin(), values.end());
    return emit(std::move(in));
}

ValueId FunctionBuilder::imageSample(uint32_t texture, uint32_t sampler, ValueId coord, uint8_t operands,
                                     std::span<const ValueId> extra, Type t) {
    Inst in = make(Op::ImageSample, t, texture);
    in.sampler = sampler;
    in.image = operands;
    in.args = {coord};
    in.args.insert(in.args.end(), extra.begin(), extra.end());
    return emit(std::move(in));
}
ValueId FunctionBuilder::imageFetch(uint32_t texture, ValueId coord, Type t) {
    Inst in = make(Op::ImageFetch, t, texture);
    in.args = {coord};
    return emit(std::move(in));
}
void FunctionBuilder::imageStore(uint32_t texture, ValueId coord, ValueId v) {
    Inst in = make(Op::ImageStore, kVoid, texture);
    in.args = {coord, v};
    emit(std::move(in));
}

void FunctionBuilder::barrier() { emit(make(Op::Barrier, kVoid)); }

void FunctionBuilder::beginIf(ValueId cond) {
    control(NodeKind::If, cond);
    open_.push_back({current().size() - 1, false, &current().back().body});
}

void FunctionBuilder::beginElse() {
    assert(!open_.empty() && !open_.back().inElse);
    Open& o = open_.back();
    o.inElse = true;
    Block& parent = open_.size() > 1 ? *open_[open_.size() - 2].inner : fn().body;
    o.inner = &parent[o.node].elseBody;
}

void FunctionBuilder::beginLoop() {
    control(NodeKind::Loop);
    open_.push_back({current().size() - 1, false, &current().back().body});
}

void FunctionBuilder::end() {
    assert(!open_.empty());
    open_.pop_back();
}

void FunctionBuilder::control(NodeKind kind, ValueId v) {
    Node n;
    n.kind = kind;
    n.value = v;
    push(std::move(n));
}
void FunctionBuilder::breakLoop() { control(NodeKind::Break); }
void FunctionBuilder::continueLoop() { control(NodeKind::Continue); }
void FunctionBuilder::discard() { control(NodeKind::Discard); }
void FunctionBuilder::ret(ValueId v) { control(NodeKind::Return, v); }

void FunctionBuilder::finish() { assert(open_.empty()); }

}  // namespace vpir
