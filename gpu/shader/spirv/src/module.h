// The parsed form of a SPIR-V module: everything the lowering needs, indexed by result id.
// Words are kept as they are; instructions are views (offset + count) into them.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "spv_enums.h"

namespace spirv {

struct Instr {
    uint16_t op = 0;
    uint16_t count = 0;   // words including the opcode word
    uint32_t at = 0;      // index of the opcode word
};

enum class TypeKind : uint8_t {
    Void, Bool, Int, Float, Vector, Matrix, Array, RuntimeArray, Struct, Pointer, Function, Image,
    Sampler, SampledImage, Other
};

struct TypeInfo {
    TypeKind kind = TypeKind::Other;
    uint32_t width = 0;        // Int/Float bits
    bool sign = false;         // Int
    uint32_t elem = 0;         // Vector/Matrix/Array/RuntimeArray element, Pointer pointee, SampledImage image
    uint32_t count = 0;        // Vector/Matrix components, Array length (resolved constant)
    uint32_t lengthId = 0;     // Array: the length constant (may be a spec constant)
    std::vector<uint32_t> members;  // Struct
    spv::Storage storage = spv::Storage::Function;  // Pointer
    // Image
    uint32_t sampledType = 0;
    spv::Dim dim = spv::Dim::Dim2D;
    uint32_t depth = 0, arrayed = 0, ms = 0, sampled = 0;
    spv::ImgFmt format = spv::ImgFmt::Unknown;
};

struct Decorations {
    std::map<uint32_t, std::vector<uint32_t>> items;  // decoration -> literal operands
    bool has(spv::Deco d) const { return items.count(uint32_t(d)) != 0; }
    uint32_t get(spv::Deco d, uint32_t fallback = 0) const {
        auto it = items.find(uint32_t(d));
        return it == items.end() || it->second.empty() ? fallback : it->second[0];
    }
};

struct Block {
    uint32_t label = 0;
    std::vector<Instr> body;      // OpPhi first, then the rest; excludes merge and terminator
    Instr merge;                  // OpSelectionMerge / OpLoopMerge, op 0 when absent
    Instr terminator;
};

struct FunctionDef {
    uint32_t id = 0, resultType = 0;
    std::vector<uint32_t> params;           // OpFunctionParameter result ids
    std::vector<uint32_t> paramTypes;
    std::vector<Block> blocks;               // first one is the entry block
    std::unordered_map<uint32_t, size_t> blockOf;  // label -> index in blocks
};

struct EntryPoint {
    spv::Model model = spv::Model::Vertex;
    uint32_t function = 0;
    std::string name;
    std::vector<uint32_t> interface;
    std::vector<Instr> modes;  // OpExecutionMode / OpExecutionModeId for this entry
};

// A constant: scalar bits, or a composite of other constants.
struct ConstantDef {
    uint32_t type = 0;
    bool composite = false;
    bool spec = false;
    bool isNull = false;
    uint64_t bits = 0;                  // scalars (bool: 0/1)
    std::vector<uint32_t> parts;        // composites
    uint32_t specId = 0xFFFFFFFFu;      // SpecId decoration, if any
    Instr specOp;                       // OpSpecConstantOp, op 0 otherwise
};

struct GlobalVar {
    uint32_t id = 0, type = 0;  // pointer type
    spv::Storage storage = spv::Storage::Private;
    uint32_t initializer = 0;
};

struct SpirvModule {
    std::vector<uint32_t> words;
    uint32_t bound = 0;
    uint32_t glslExt = 0;  // the GLSL.std.450 import id, 0 if absent
    std::vector<EntryPoint> entries;
    std::unordered_map<uint32_t, TypeInfo> types;
    std::unordered_map<uint32_t, ConstantDef> constants;
    std::unordered_map<uint32_t, GlobalVar> globals;
    std::vector<uint32_t> globalOrder;
    std::unordered_map<uint32_t, FunctionDef> functions;
    std::unordered_map<uint32_t, Decorations> decorations;
    std::map<std::pair<uint32_t, uint32_t>, Decorations> memberDecorations;
    std::unordered_map<uint32_t, uint32_t> resultType;   // id -> type id, for every typed result
    std::unordered_map<uint32_t, std::string> names;     // OpName, for diagnostics

    uint32_t w(const Instr& in, uint32_t i) const { return words[in.at + i]; }
    const TypeInfo* type(uint32_t id) const {
        auto it = types.find(id);
        return it == types.end() ? nullptr : &it->second;
    }
    const Decorations* deco(uint32_t id) const {
        auto it = decorations.find(id);
        return it == decorations.end() ? nullptr : &it->second;
    }
    const Decorations* memberDeco(uint32_t structId, uint32_t member) const {
        auto it = memberDecorations.find({structId, member});
        return it == memberDecorations.end() ? nullptr : &it->second;
    }
    std::string nameOf(uint32_t id) const {
        auto it = names.find(id);
        return it == names.end() ? "%" + std::to_string(id) : it->second + "(%" + std::to_string(id) + ")";
    }
};

std::string literalString(const SpirvModule& m, const Instr& in, uint32_t firstWord, uint32_t* wordsUsed = nullptr);

}  // namespace spirv
