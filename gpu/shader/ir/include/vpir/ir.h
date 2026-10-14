// VPIR: VPKEngine's shader IR. Typed SSA values inside a structured control-flow tree.
//
// Values that must cross control-flow regions go through function-local variables (there is no
// phi), so every region maps directly onto a C-like block in the target language.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpir {

// ---- Types ---------------------------------------------------------------------------------------

enum class Scalar : uint8_t { Void, Bool, I32, U32, F16, F32 };

struct Type {
    Scalar scalar = Scalar::Void;
    uint8_t width = 1;  // 1 = scalar, 2..4 = vector

    constexpr Type() = default;
    constexpr Type(Scalar s, uint8_t w = 1) : scalar(s), width(w) {}
    constexpr bool operator==(const Type&) const = default;

    constexpr bool isVoid() const { return scalar == Scalar::Void; }
    constexpr bool isBool() const { return scalar == Scalar::Bool; }
    constexpr bool isFloat() const { return scalar == Scalar::F16 || scalar == Scalar::F32; }
    constexpr bool isInt() const { return scalar == Scalar::I32 || scalar == Scalar::U32; }
    constexpr bool isNumeric() const { return isFloat() || isInt(); }
    constexpr bool isScalar() const { return width == 1 && !isVoid(); }
    constexpr Type element() const { return {scalar, 1}; }
    constexpr Type withWidth(uint8_t w) const { return {scalar, w}; }
    constexpr Type withScalar(Scalar s) const { return {s, width}; }
    // Size in bytes; bool has no defined memory size and reports 0.
    constexpr unsigned byteSize() const {
        unsigned e = scalar == Scalar::F16 ? 2 : (isBool() || isVoid()) ? 0 : 4;
        return e * width;
    }
};

inline constexpr Type kVoid{};
inline constexpr Type kBool{Scalar::Bool};
inline constexpr Type kI32{Scalar::I32};
inline constexpr Type kU32{Scalar::U32};
inline constexpr Type kF16{Scalar::F16};
inline constexpr Type kF32{Scalar::F32};

std::string toString(Type t);

// ---- Module-level declarations -------------------------------------------------------------------

enum class Stage : uint8_t { Vertex, Fragment, Compute, Object, Mesh };

enum class Builtin : uint8_t {
    None,
    Position,             // vertex/mesh out, fragment in: vec4<f32>
    PointSize,            // vertex/mesh out: f32
    VertexId,             // vertex in: u32
    InstanceId,           // vertex in: u32
    FrontFacing,          // fragment in: bool
    SampleId,             // fragment in: u32
    SampleMask,           // fragment in/out: u32
    FragDepth,            // fragment out: f32
    PrimitiveId,          // fragment in: u32
    Layer,                // vertex/mesh out, fragment in: u32 (render target array index)
    GlobalInvocationId,   // compute/object/mesh in: vec3<u32>
    LocalInvocationId,    // compute/object/mesh in: vec3<u32>
    WorkgroupId,          // compute/object/mesh in: vec3<u32>
    NumWorkgroups,        // compute/object/mesh in: vec3<u32>
    LocalInvocationIndex, // compute/object/mesh in: u32
    SubgroupInvocation,   // fragment/compute/object/mesh in: u32
    SubgroupSize,         // fragment/compute/object/mesh in: u32
    ClipDistance,         // vertex out: f32, one per distance; IoVar::location = its index (< 8)
};

enum class Interp : uint8_t { Perspective, Linear, Flat };
enum class Sampling : uint8_t { Center, Centroid, Sample };

// A stage input or output. Exactly one of `builtin` / `location` is meaningful (both for
// ClipDistance: location is the distance index).
struct IoVar {
    std::string name;
    Type type;
    Builtin builtin = Builtin::None;
    uint32_t location = 0;
    Interp interp = Interp::Perspective;   // fragment inputs, vertex/mesh outputs
    Sampling sampling = Sampling::Center;
    bool perPrimitive = false;              // mesh outputs only
    uint8_t index = 0;                      // fragment outputs: dual-source blend index (0 or 1)
};

// Buffer: byte-addressed device memory. ConstantBuffer: read-only, byte-addressed, uniform across the
// draw; the emitter places it in the `constant` address space so reads skip generic memory paths.
// BufferSize: the byte size of the range bound to Buffer slot `index` (the same binding), read with
// buffer_size (SPIR-V OpArrayLength). Last, so the other kinds keep their argument-table order.
enum class ResourceKind : uint8_t { Buffer, ConstantBuffer, Texture, Sampler, StorageTexture, BufferSize };
enum class TexDim : uint8_t { D1, D2, D2Array, D3, Cube, CubeArray };
enum class Access : uint8_t { Read, Write, ReadWrite };

// A resource slot. Buffer addresses are byte offsets and always 4-byte aligned.
struct Resource {
    std::string name;
    ResourceKind kind = ResourceKind::Buffer;
    uint32_t index = 0;               // slot index, unique per kind
    TexDim dim = TexDim::D2;          // textures
    Scalar elem = Scalar::F32;        // textures: component type
    bool depth = false;               // sampled textures: depth texture (compare sampling)
    bool multisample = false;         // sampled textures, d2/d2_array: fetch only, lod operand = sample
    Access access = Access::ReadWrite;  // buffers, storage textures
};

enum class Topology : uint8_t { Point, Line, Triangle };

// ---- Ops -----------------------------------------------------------------------------------------
//
// Operand signature characters (shared by the parser, printer and verifier):
//   v  value                      *  remaining values (variadic)
//   $  variable                   i  input            o  output
//   r  resource                   s  sampler resource f  function
//   m  memory target (buffer resource, `shared`, `payload` or `private`)
//   n  one integer literal        N  integer literal list
//   k  constant literal list (typed by the result)    I  optional image operands
//
// Semantics that are not obvious from the name:
//   shl/shr        shift amount must be < bit width (frontends mask); shr is arithmetic on i32
//   rem            truncated remainder (fmod for floats)
//   round          to nearest even
//   find_lsb       count of trailing zeros (bit width for 0); find_msb: index of the highest set
//                  bit, all ones for 0
//   select c,a,b   c ? a : b (c is scalar bool or a bool vector of the operands' width)
//   mix a,b,t      a + (b - a) * t
//   load/store     m, byte address (u32), [value]; 32-bit scalars or vectors
//   atomic_*       m, byte address, value -> previous value
//   atomic_compare_exchange  m, address, comparator, value -> previous value
//   image_atomic_*  storage texture (read_write, i32/u32 texels, not cube), integer coord (with the
//                  layer for arrays), value[, value] -> previous texel; as atomic_* otherwise
//   image_gather   texture, sampler, component, coord
//   image_query_lod  texture, sampler, coord (with the layer for arrays) -> vec2<f32>(lod a sample
//                  would use, clamped to the view and sampler; lod computed from derivatives,
//                  unclamped). Fragment-only.
//   interp_offset  offset from the pixel centre, in pixels
//   barrier        workgroup execution + shared memory; barrier_all also orders device memory
//   subgroup_*     defined over a wave of Module::waveSize lanes (32 or 64): lane ids run
//                  0..waveSize-1, subgroup_ballot sets bit i of its vec4<u32> mask for lane i (bits
//                  >= waveSize are 0) and broadcast/shuffle lanes must be < waveSize. A wave is the
//                  set of invocations that execute in lockstep; it is not tied to the workgroup.
//   quad_shuffle v, lane  the value of `v` in lane `lane` (0..3) of this invocation's quad: the
//                  2x2 pixels a fragment shares derivatives with (lane = x + 2y), or four
//                  consecutive invocations in compute. quad_any: any lane of the quad has it true.
//                  Independent of the wave size.
//   framebuffer_fetch n  fragment only: the current value of colour attachment n (0..7) at this
//                  fragment's pixel, for the sample being shaded when the shader runs per sample
//                  (Metal has no explicit-sample read, so there is no sample operand). Every use of
//                  one index has the same type, which matches the attachment's format class (f16/f32
//                  for float and normalized formats, i32/u32 for integer ones) and an output at that
//                  location if there is one. Depth/stencil attachments cannot be fetched.
//   mesh_vertex/mesh_primitive  index, then one value per per-vertex/per-primitive output in order
#define VPIR_OPS(X)                                                                                \
    X(Const, "const", "k")                                                                         \
    X(LoadVar, "load_var", "$")                                                                    \
    X(StoreVar, "store_var", "$v")                                                                 \
    X(LoadInput, "load_input", "i")                                                                \
    X(StoreOutput, "store_output", "ov")                                                           \
    X(Call, "call", "f*")                                                                          \
    X(Neg, "neg", "v")                                                                             \
    X(Abs, "abs", "v")                                                                             \
    X(Not, "not", "v")                                                                             \
    X(Add, "add", "vv")                                                                            \
    X(Sub, "sub", "vv")                                                                            \
    X(Mul, "mul", "vv")                                                                            \
    X(Div, "div", "vv")                                                                            \
    X(Rem, "rem", "vv")                                                                            \
    X(Min, "min", "vv")                                                                            \
    X(Max, "max", "vv")                                                                            \
    X(And, "and", "vv")                                                                            \
    X(Or, "or", "vv")                                                                              \
    X(Xor, "xor", "vv")                                                                            \
    X(Shl, "shl", "vv")                                                                            \
    X(Shr, "shr", "vv")                                                                            \
    X(Fma, "fma", "vvv")                                                                           \
    X(Clamp, "clamp", "vvv")                                                                       \
    X(Mix, "mix", "vvv")                                                                           \
    X(Eq, "eq", "vv")                                                                              \
    X(Ne, "ne", "vv")                                                                              \
    X(Lt, "lt", "vv")                                                                              \
    X(Le, "le", "vv")                                                                              \
    X(Gt, "gt", "vv")                                                                              \
    X(Ge, "ge", "vv")                                                                              \
    X(IsNan, "isnan", "v")                                                                         \
    X(IsInf, "isinf", "v")                                                                         \
    X(Select, "select", "vvv")                                                                     \
    X(Convert, "convert", "v")                                                                     \
    X(Bitcast, "bitcast", "v")                                                                     \
    X(BitfieldExtract, "bitfield_extract", "vvv")                                                  \
    X(BitfieldInsert, "bitfield_insert", "vvvv")                                                   \
    X(BitCount, "bit_count", "v")                                                                  \
    X(BitReverse, "bit_reverse", "v")                                                              \
    X(FindLsb, "find_lsb", "v")                                                                    \
    X(FindMsb, "find_msb", "v")                                                                    \
    X(Floor, "floor", "v")                                                                         \
    X(Ceil, "ceil", "v")                                                                           \
    X(Trunc, "trunc", "v")                                                                         \
    X(Round, "round", "v")                                                                         \
    X(Fract, "fract", "v")                                                                         \
    X(Sqrt, "sqrt", "v")                                                                           \
    X(Rsq, "rsq", "v")                                                                             \
    X(Rcp, "rcp", "v")                                                                             \
    X(Exp2, "exp2", "v")                                                                           \
    X(Log2, "log2", "v")                                                                           \
    X(Sin, "sin", "v")                                                                             \
    X(Cos, "cos", "v")                                                                             \
    X(Saturate, "saturate", "v")                                                                   \
    X(Pow, "pow", "vv")                                                                            \
    X(Dot, "dot", "vv")                                                                            \
    X(Length, "length", "v")                                                                       \
    X(Normalize, "normalize", "v")                                                                 \
    X(Construct, "construct", "*")                                                                 \
    X(Extract, "extract", "vn")                                                                    \
    X(Insert, "insert", "vvn")                                                                     \
    X(Shuffle, "shuffle", "vvN")                                                                   \
    X(Load, "load", "mv")                                                                          \
    X(Store, "store", "mvv")                                                                       \
    X(AtomicAdd, "atomic_add", "mvv")                                                              \
    X(AtomicSub, "atomic_sub", "mvv")                                                              \
    X(AtomicMin, "atomic_min", "mvv")                                                              \
    X(AtomicMax, "atomic_max", "mvv")                                                              \
    X(AtomicAnd, "atomic_and", "mvv")                                                              \
    X(AtomicOr, "atomic_or", "mvv")                                                                \
    X(AtomicXor, "atomic_xor", "mvv")                                                              \
    X(AtomicExchange, "atomic_exchange", "mvv")                                                    \
    X(AtomicCompareExchange, "atomic_compare_exchange", "mvvv")                                    \
    X(ImageAtomicAdd, "image_atomic_add", "rvv")                                                   \
    X(ImageAtomicSub, "image_atomic_sub", "rvv")                                                   \
    X(ImageAtomicMin, "image_atomic_min", "rvv")                                                   \
    X(ImageAtomicMax, "image_atomic_max", "rvv")                                                   \
    X(ImageAtomicAnd, "image_atomic_and", "rvv")                                                   \
    X(ImageAtomicOr, "image_atomic_or", "rvv")                                                     \
    X(ImageAtomicXor, "image_atomic_xor", "rvv")                                                   \
    X(ImageAtomicExchange, "image_atomic_exchange", "rvv")                                         \
    X(ImageAtomicCompareExchange, "image_atomic_compare_exchange", "rvvv")                         \
    X(ImageSample, "image_sample", "rsvI")                                                         \
    X(ImageGather, "image_gather", "rsnvI")                                                        \
    X(ImageFetch, "image_fetch", "rvI")                                                            \
    X(ImageStore, "image_store", "rvv")                                                            \
    X(ImageSize, "image_size", "rI")                                                               \
    X(ImageLevels, "image_levels", "r")                                                            \
    X(ImageSamples, "image_samples", "r")                                                          \
    X(BufferSize, "buffer_size", "r")                                                              \
    X(ImageQueryLod, "image_query_lod", "rsv")                                                     \
    X(InterpCentroid, "interp_centroid", "i")                                                      \
    X(InterpSample, "interp_sample", "iv")                                                         \
    X(InterpOffset, "interp_offset", "iv")                                                         \
    X(FramebufferFetch, "framebuffer_fetch", "n")                                                  \
    X(Ddx, "ddx", "v")                                                                             \
    X(Ddy, "ddy", "v")                                                                             \
    X(Fwidth, "fwidth", "v")                                                                       \
    X(Barrier, "barrier", "")                                                                      \
    X(BarrierAll, "barrier_all", "")                                                               \
    X(SubgroupBallot, "subgroup_ballot", "v")                                                      \
    X(SubgroupBroadcast, "subgroup_broadcast", "vv")                                               \
    X(SubgroupShuffle, "subgroup_shuffle", "vv")                                                   \
    X(SubgroupReadFirst, "subgroup_read_first", "v")                                               \
    X(SubgroupAny, "subgroup_any", "v")                                                            \
    X(SubgroupAll, "subgroup_all", "v")                                                            \
    X(QuadShuffle, "quad_shuffle", "vv")                                                           \
    X(QuadAny, "quad_any", "v")                                                                    \
    X(MeshDispatch, "mesh_dispatch", "vvv")                                                        \
    X(MeshVertex, "mesh_vertex", "v*")                                                             \
    X(MeshPrimitive, "mesh_primitive", "v*")                                                       \
    X(MeshIndex, "mesh_index", "vv")                                                               \
    X(MeshPrimitiveCount, "mesh_primitive_count", "v")

enum class Op : uint8_t {
#define VPIR_ENUM(id, name, sig) id,
    VPIR_OPS(VPIR_ENUM)
#undef VPIR_ENUM
};

struct OpInfo {
    const char* name;
    const char* signature;
};
const OpInfo& opInfo(Op op);
bool opFromName(std::string_view name, Op& out);

// Optional image operands; their values follow the required arguments in this bit order.
enum ImageOperand : uint8_t {
    kImageLod = 1 << 0,     // 1 value: f32 for sampling, integer mip for fetch/size
    kImageBias = 1 << 1,    // 1 value: f32
    kImageGrad = 1 << 2,    // 2 values: d/dx, d/dy
    kImageOffset = 1 << 3,  // 1 value: integer texel offset
    kImageCompare = 1 << 4, // 1 value: f32 depth reference
};

// ---- Functions -----------------------------------------------------------------------------------

using ValueId = uint32_t;
inline constexpr ValueId kNoValue = 0xFFFFFFFFu;

// Memory targets that are not buffer resources.
inline constexpr uint32_t kTargetShared = 0xFFFFFFFEu;
inline constexpr uint32_t kTargetPayload = 0xFFFFFFFDu;
// Per-invocation memory (any stage; no atomics): `private` in the text form.
inline constexpr uint32_t kTargetPrivate = 0xFFFFFFFCu;
// Per-variable per-invocation memory: kTargetPrivateSlot0 + n is slot n of Module::privateSlots
// (`private.n` in the text form). Each slot is its own array in the backend, so one dynamically
// indexed variable cannot force every other variable of the shader out of registers.
inline constexpr uint32_t kTargetPrivateSlot0 = 0xF0000000u;
inline constexpr uint32_t kMaxPrivateSlots = 0x0FFFFF00u;
inline constexpr bool isPrivateSlot(uint32_t t) { return t >= kTargetPrivateSlot0 && t - kTargetPrivateSlot0 < kMaxPrivateSlots; }
inline constexpr uint32_t privateSlotIndex(uint32_t t) { return t - kTargetPrivateSlot0; }

struct Inst {
    Op op = Op::Const;
    Type type;                   // result type; void when there is no result
    ValueId result = kNoValue;
    bool precise = false;        // forbid fast-math relaxations for this instruction
    std::vector<ValueId> args;   // value operands, in signature order
    uint32_t sym = 0;            // variable / input / output / resource / function / memory target
    uint32_t sampler = 0;        // sampler resource for image_sample / image_gather
    std::vector<uint32_t> lits;  // constant bits (one per component) or integer literals
    uint8_t image = 0;           // ImageOperand mask
};

enum class NodeKind : uint8_t { Inst, If, Loop, Break, Continue, Return, Discard };

struct Node;
using Block = std::vector<Node>;

// One element of the structured control-flow tree. `loop` repeats `body` until a `break`.
struct Node {
    NodeKind kind = NodeKind::Inst;
    Inst inst;                  // Inst
    ValueId value = kNoValue;   // If: condition; Return: returned value (or none)
    Block body;                 // If: then-branch; Loop: body
    Block elseBody;             // If: else-branch
};

struct Variable {
    Type type;
};

struct Function {
    std::string name;
    std::vector<ValueId> params;
    Type returnType;
    std::vector<Variable> vars;
    Block body;
    // Type of each value id defined in the function (params and results); void = unused id.
    std::vector<Type> valueTypes;

    Type typeOf(ValueId v) const { return v < valueTypes.size() ? valueTypes[v] : kVoid; }
};

struct Module {
    Stage stage = Stage::Compute;
    std::vector<IoVar> inputs;
    std::vector<IoVar> outputs;
    std::vector<Resource> resources;
    uint32_t workgroupSize[3] = {1, 1, 1};  // compute / object / mesh
    uint32_t sharedBytes = 0;                // workgroup-shared memory (compute / object / mesh)
    uint32_t payloadBytes = 0;               // object -> mesh payload
    uint32_t privateBytes = 0;               // per-invocation memory (kTargetPrivate)
    std::vector<uint32_t> privateSlots;      // byte size (multiple of 16) of each private slot
    uint32_t maxVertices = 0;                // mesh
    uint32_t maxPrimitives = 0;              // mesh
    Topology topology = Topology::Triangle;  // mesh
    // Lanes per wave for subgroup ops: 32 or 64 (see the subgroup_* notes above). GCN modules are
    // 64; Apple SIMD groups are 32, so emitters may require the frontend to lower wave-64 subgroup
    // code (the MSL emitter does).
    uint32_t waveSize = 32;
    // Helpers first; a function may only call functions declared before it. The entry point is
    // the function named "main" (no parameters, void).
    std::vector<Function> functions;

    const Function* entry() const;
};

const char* toString(Stage s);
const char* toString(Builtin b);
const char* toString(Interp i);
const char* toString(Sampling s);
const char* toString(ResourceKind k);
const char* toString(TexDim d);
const char* toString(Access a);
const char* toString(Topology t);

// Number of coordinate components of a texture dimension, excluding the array layer.
unsigned coordWidth(TexDim d);
bool isArrayed(TexDim d);
unsigned verticesPerPrimitive(Topology t);

// Builtin metadata used by the verifier and the emitter.
struct BuiltinInfo {
    Type type;
    bool input;   // true = stage input, false = stage output
    unsigned stages;  // bit mask of (1 << Stage)
};
BuiltinInfo builtinInfo(Builtin b, bool asInput);

// f16 <-> f32 bit conversions (round to nearest even).
uint16_t floatToHalf(float f);
float halfToFloat(uint16_t h);

}  // namespace vpir
