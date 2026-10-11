// Builder for VPIR functions. Instructions are appended to the innermost open region; regions are
// opened with beginIf/beginElse/beginLoop and closed with end().
//
//   FunctionBuilder b(module, "main");
//   ValueId uv = b.loadInput(0, vec(kF32, 2));
//   ValueId c = b.binary(Op::Lt, x, y, kBool);
//   b.beginIf(c); b.discard(); b.end();
//   b.finish();
#pragma once

#include <initializer_list>
#include <span>

#include "vpir/ir.h"

namespace vpir {

constexpr Type vec(Type scalar, uint8_t width) { return scalar.withWidth(width); }

class FunctionBuilder {
public:
    // Starts a new function at the end of module.functions. The module must outlive the builder and
    // must not have functions added by anyone else until finish().
    FunctionBuilder(Module& module, std::string name, Type returnType = kVoid);

    ValueId param(Type t);
    uint32_t variable(Type t);

    // Generic append; returns the result id (kNoValue when `type` is void).
    ValueId emit(Inst inst);

    ValueId constant(Type t, std::initializer_list<uint32_t> bits);
    ValueId constF32(float v);
    ValueId constU32(uint32_t v);
    ValueId constI32(int32_t v);
    ValueId constBool(bool v);

    ValueId unary(Op op, ValueId a, Type t, bool precise = false);
    ValueId binary(Op op, ValueId a, ValueId b, Type t, bool precise = false);
    ValueId ternary(Op op, ValueId a, ValueId b, ValueId c, Type t, bool precise = false);
    ValueId nary(Op op, std::span<const ValueId> args, Type t, bool precise = false);

    ValueId loadVar(uint32_t var);
    void storeVar(uint32_t var, ValueId v);
    ValueId loadInput(uint32_t input, Type t);
    void storeOutput(uint32_t output, ValueId v);
    ValueId call(uint32_t function, std::span<const ValueId> args, Type t);

    ValueId extract(ValueId v, uint32_t index, Type t);
    ValueId construct(std::span<const ValueId> parts, Type t);

    ValueId load(uint32_t target, ValueId address, Type t);
    void store(uint32_t target, ValueId address, ValueId v);
    ValueId atomic(Op op, uint32_t target, ValueId address, std::span<const ValueId> values, Type t);

    // `extra` holds the optional image operand values in ImageOperand bit order.
    ValueId imageSample(uint32_t texture, uint32_t sampler, ValueId coord, uint8_t operands,
                        std::span<const ValueId> extra, Type t);
    ValueId imageFetch(uint32_t texture, ValueId coord, Type t);
    void imageStore(uint32_t texture, ValueId coord, ValueId v);

    void barrier();

    void beginIf(ValueId cond);
    void beginElse();
    void beginLoop();
    void end();

    void breakLoop();
    void continueLoop();
    void discard();
    void ret(ValueId v = kNoValue);

    // Closes the function; every region must have been ended.
    void finish();

private:
    Function& fn();
    Block& current();
    ValueId newValue(Type t);
    void push(Node n);
    void control(NodeKind kind, ValueId v = kNoValue);

    Module& module_;
    size_t index_;
    // Path from the function body to the innermost open region: node index + which branch.
    struct Open {
        size_t node;
        bool inElse;
    };
    std::vector<Open> open_;
};

}  // namespace vpir
