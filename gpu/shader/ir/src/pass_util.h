// Internal helpers shared by the VPIR passes.
#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "vpir/ir.h"

namespace vpir {

// Calls f on every instruction of a block tree, in order.
inline void forEachInst(Block& b, const std::function<void(Inst&)>& f) {
    for (Node& n : b) {
        if (n.kind == NodeKind::Inst) f(n.inst);
        forEachInst(n.body, f);
        forEachInst(n.elseBody, f);
    }
}

// Result id -> defining instruction. Pointers stay valid until the block tree is edited.
inline std::unordered_map<ValueId, const Inst*> definitions(Block& b) {
    std::unordered_map<ValueId, const Inst*> defs;
    forEachInst(b, [&](Inst& in) {
        if (in.result != kNoValue) defs[in.result] = &in;
    });
    return defs;
}

// Removes instructions whose result is unused and that have no effect besides it. Calls are kept
// (a helper may never return); atomics are kept for their store.
void removeDeadValues(Function& f);

}  // namespace vpir
