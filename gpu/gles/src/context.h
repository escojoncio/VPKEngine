// The GLES state machine on Vulkan (the model ANGLE's Vulkan backend and Zink use: GL state turned
// into cached Vulkan objects at draw time), own code.
//
// Model
//  * One VkDevice per process (Device), shared by every EGL context. Each context records into its
//    own command buffer; it is submitted at eglSwapBuffers, glFlush/glFinish, fences, readbacks and
//    when another context becomes current on the thread.
//  * Orientation: GL's row 0 (bottom) is memory row 0 everywhere (textures, renderbuffers, the
//    default framebuffer). With that, GL viewports/scissors/window coordinates map to Vulkan's
//    unchanged and glReadPixels needs no flip; only the winding flips (GL CCW = Vulkan CW) and
//    whoever presents the default framebuffer or a GL swapchain image shows it bottom-up.
//  * Buffers are host-visible (unified memory). A write to a buffer the GPU may still read
//    (submitted or recorded work) gives the GL buffer new storage first (copy on write), so GL's
//    "commands see the data as of their call" holds without stalls.
//  * Default-block uniforms live on the CPU per program (std140) and are copied into a per-submit
//    ring at draws that need them (dynamic uniform buffer offset).
//  * Every image stays in VK_IMAGE_LAYOUT_GENERAL; transfers and passes are ordered by full
//    pipeline barriers. Simple and correct; VPVulkan derives Metal hazards itself.
#pragma once

#include <thread>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "formats.h"
#include "gl_enums.h"
#include "gl_enums31.h"
#include "glsl.h"
#include "vk.h"

namespace vpgles {

void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// One `gles.glsl.lenient rule=... <what>=<name>` line per rule a shader (what "shader") or program ("program") broke that
// strict GLSL (VPGLES_STRICT_GLSL=1) would have rejected and the default lenient mode let through.
void logLenient(const std::vector<std::string>& rules, const char* what, unsigned name);

// Diagnostics: textures defined per sized internal format ("gles.tex fmt=0x.... count=N"); texStatsTick prints the
// changed counts every 10 s (force: now, at eglTerminate / device shutdown; called from eglSwapBuffers and vpgles_flush),
// and the first texture of a format prints at once; at most 256 lines per process.
void texStatsCount(GLenum internalFormat);
void texStatsTick(bool force);

// Diagnostics: "gles.pso.recipe saved= loaded= prewarmed= miss_in_game=" (VPVulkan's pipeline-recipe store, see vpvk_recipe_stats):
// printed when the counters changed and 10 s passed since the last line (force: now, even unchanged, at eglTerminate / device
// shutdown; called from eglSwapBuffers, vpgles_flush and the creation of a new pipeline). Needs Device::lock.
void recipeStatsTick(bool force);

constexpr int kMaxTextureUnits = 96;  // Quest 3 (Adreno 740): GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS
constexpr int kMaxVertexAttribs = 32;  // Quest 3: GL_MAX_VERTEX_ATTRIBS
constexpr int kHwVertexAttribs = 31;   // VPVulkan maxVertexInputAttributes (Metal vertex buffer table): locations above are dropped
constexpr int kMaxDrawBuffers = 8;
constexpr int kMaxUniformBufferBindings = 84;  // Quest 3: GL_MAX_UNIFORM_BUFFER_BINDINGS
constexpr int kMaxVertexBindings = kMaxVertexAttribs;
constexpr int kMaxShaderStorageBindings = 36;  // Quest 3: GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
constexpr int kMaxAtomicCounterBindings = 8;
constexpr int kMaxTransformFeedbackBindings = 4;
constexpr int kMaxImageUnits = 8;
static_assert(kMaxTextureUnits == vpglsl::kBindingSamplers && kMaxImageUnits == vpglsl::kBindingImages && kMaxAtomicCounterBindings == vpglsl::kBindingAtomics &&
                  kMaxUniformBufferBindings == vpglsl::kBindingUniform && kMaxShaderStorageBindings == vpglsl::kBindingStorage,
              "GLSL binding limits and GL limits must agree");

// ---- device ---------------------------------------------------------------------------------------
struct Device;

// Host-visible memory + buffer (one allocation per object: unified memory, cheap).
struct BufferStorage {
    Device* dev = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* host = nullptr;
    VkDeviceSize size = 0;
    uint64_t lastUse = 0;  // serial of the last submission that references it (0 = never)
    ~BufferStorage();
};
using StorageP = std::shared_ptr<BufferStorage>;

struct Device {
    Vk vk;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDeviceProperties props{};
    VkCommandPool pool = VK_NULL_HANDLE;
    std::recursive_mutex lock;  // the global GL lock (every entry point holds it)

    // Submission serials: the n-th vkQueueSubmit has serial n; `completed` is the highest one whose
    // fence signalled.
    uint64_t submitted = 0, completed = 0;
    struct InFlight {
        uint64_t serial;
        VkFence fence;
        VkCommandBuffer cb;
        std::vector<std::shared_ptr<void>> keep;  // objects to keep alive until it completes
        VkDescriptorPool descPool = VK_NULL_HANDLE;
    };
    std::vector<InFlight> inFlight;
    std::vector<VkDescriptorPool> freePools;

    bool init(PFN_vkGetInstanceProcAddr gipa);
    void shutdown();
    int memoryType(uint32_t bits, VkMemoryPropertyFlags want) const;
    StorageP newStorage(VkDeviceSize size, VkBufferUsageFlags usage);
    void retire(bool wait);         // polls (or waits for) in-flight fences
    void waitSerial(uint64_t serial);
    VkDescriptorPool takePool();
};

Device* device();  // the process device (null before vpgles_init)

// ---- objects --------------------------------------------------------------------------------------
struct Buffer {
    GLuint name = 0;
    bool bound = false;  // bound once: only then glIsBuffer says it is a buffer object
    StorageP storage;
    GLsizeiptr size = 0;
    GLenum usage = gl::STATIC_DRAW;
    // Mapping (glMapBufferRange)
    bool mapped = false;
    GLbitfield mapAccess = 0;
    GLintptr mapOffset = 0;
    GLsizeiptr mapLength = 0;
};

struct SamplerParams {
    GLenum minFilter = gl::NEAREST_MIPMAP_LINEAR, magFilter = gl::LINEAR;
    GLenum wrapS = gl::REPEAT, wrapT = gl::REPEAT, wrapR = gl::REPEAT;
    float minLod = -1000.0f, maxLod = 1000.0f, maxAnisotropy = 1.0f;
    GLenum compareMode = gl::NONE, compareFunc = gl::LEQUAL;
    float border[4] = {0, 0, 0, 0};
    int32_t borderI[4] = {0, 0, 0, 0};  // glTexParameterI[u]iv: the border colour as the pure integers the app gave
    bool borderInt = false;             // the last write of the border colour was an integer one (borderI is the exact value)
    bool operator==(const SamplerParams& o) const;
};

struct Image {  // a VkImage with its memory
    Device* dev = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool external = false;  // not owned (swapchain / surface image created elsewhere)
    VkImageType type = VK_IMAGE_TYPE_2D;
    const FormatInfo* format = nullptr;
    uint32_t width = 0, height = 0, depth = 1, levels = 1, layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    bool cube = false;
    bool storage = false;  // created with STORAGE_BIT (only textures used with glBindImageTexture pay for it)
    std::map<uint64_t, VkImageView> views;  // cached views
    // glInvalidateFramebuffer: attachment views (and aspects: 1 colour, 2 depth, 4 stencil) whose contents are undefined.
    // The next pass that begins on one of them loads DONT_CARE; any other use of the image defines it again.
    std::map<VkImageView, uint8_t> invalid;
    ~Image();
    VkImageAspectFlags aspect() const;
    // A 2D (array) view of one level and `layerCount` layers from `layer`, for attachments.
    VkImageView attachmentView(uint32_t level, uint32_t layer, uint32_t layerCount);
    // A view for a storage image descriptor: `type` 2D / 2D_ARRAY / 3D over one level and `layerCount` layers.
    // `format` (when not UNDEFINED and not the image's own): the texel format the view reinterprets the image with
    // (glBindImageTexture with an image format other than the texture's, of the same size).
    VkImageView storageView(uint32_t level, uint32_t layer, uint32_t layerCount, VkImageViewType type,
                            VkFormat format = VK_FORMAT_UNDEFINED);
};
using ImageP = std::shared_ptr<Image>;
// A 2D (array) image of `samples` samples with its device-local memory bound; null when Vulkan refuses it. The caller
// records Context::imageInit.
ImageP createImage2D(Device* dev, const FormatInfo* f, uint32_t w, uint32_t h, uint32_t layers, uint32_t samples,
                     VkImageUsageFlags usage, VkImageCreateFlags flags = 0);

struct TexelBufferView;

struct Texture {
    GLuint name = 0;
    GLenum target = 0;  // 0 until first bound
    ImageP image;
    bool immutable = false;
    bool wantStorage = false;  // glBindImageTexture used it: its image is (re)created with STORAGE_BIT
    GLint immutableLevels = 0;
    // Level definitions (glTexImage*): per level, the size and format the app defined.
    struct Level {
        uint32_t w = 0, h = 0, d = 0;
        const FormatInfo* format = nullptr;
    };
    std::vector<Level> levels;
    // Cube maps: the same definitions per face (a face's level can differ from the others', which makes the cube incomplete).
    std::array<std::vector<Level>, 6> faceLevels;
    // Uploads to levels above 0 that arrived before level 0 defined the image (UE4 streams the small mips first): kept
    // here (already converted, ready to copy) until the base creates the image, then uploaded and freed.
    struct PendingUpload {
        uint32_t level, layer;
        int32_t x, y, z;
        uint32_t w, h, d;
        std::vector<uint8_t> data;
    };
    std::vector<PendingUpload> pending;
    size_t pendingBytes = 0;
    SamplerParams sampler;
    GLint baseLevel = 0, maxLevel = 1000;
    GLenum swizzle[4] = {gl::RED, gl::GREEN, gl::BLUE, gl::ALPHA};
    GLenum depthStencilMode = gl::DEPTH_COMPONENT;
    bool srgbDecode = true;
    GLint samples = 0;          // glTexStorage2DMultisample / glTexImage2DMultisample (the image itself is 1x)
    bool fixedSampleLocations = false;
    // Buffer texture (glTexBuffer / glTexBufferRange): the data store is a range of a buffer object.
    GLuint bufferName = 0;
    GLintptr bufferOffset = 0;
    GLsizeiptr bufferSize = 0;  // bytes of the range
    const FormatInfo* bufferFormat = nullptr;
    std::shared_ptr<TexelBufferView> bufferView;  // lazily created, replaced when the range / storage changes
    // External OES image (EGLImage / video): sampled from this image.
    VkImageView sampledView();
    uint64_t viewKey() const;
};

// A VkBufferView over a range of a buffer's storage (texel buffer descriptor of a samplerBuffer).
struct TexelBufferView {
    Device* dev = nullptr;
    VkBufferView view = VK_NULL_HANDLE;
    StorageP storage;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkDeviceSize offset = 0, range = 0;
    ~TexelBufferView();
};

struct Renderbuffer {
    GLuint name = 0;
    bool bound = false;  // glIsRenderbuffer
    ImageP image;
    GLenum internalFormat = gl::RGBA4;
    GLsizei width = 0, height = 0, samples = 0;
};

struct Attachment {
    enum Kind : uint8_t { None, Tex, Rbo } kind = None;
    GLuint name = 0;
    GLint level = 0, layer = 0;     // layer: cube face / array layer
    GLint numViews = 0, baseView = 0;  // OVR_multiview
    bool operator==(const Attachment& o) const {
        return kind == o.kind && name == o.name && level == o.level && layer == o.layer && numViews == o.numViews &&
               baseView == o.baseView;
    }
};

struct Framebuffer {
    GLuint name = 0;
    bool bound = false;  // glIsFramebuffer
    std::array<Attachment, kMaxDrawBuffers> color{};
    Attachment depth, stencil;
    std::array<GLenum, kMaxDrawBuffers> drawBuffers{};
    GLenum readBuffer = gl::COLOR_ATTACHMENT0;
    // glFramebufferParameteri (ES 3.1): the size an attachment-less framebuffer has, and the rest as stored state.
    GLint defaultWidth = 0, defaultHeight = 0, defaultLayers = 0, defaultSamples = 0;
    bool defaultFixedSampleLocations = false;
    Framebuffer() {
        drawBuffers.fill(gl::NONE);
        drawBuffers[0] = gl::COLOR_ATTACHMENT0;
    }
};

// Program pipeline object (ES 3.1, not shared between contexts): the program each stage runs. A stage
// program is used only when it holds that stage; the GLSL linker needs vertex and fragment together, so a
// draw works when both stages are the same program (see Context::graphicsProgram).
struct ProgramPipeline {
    GLuint name = 0;
    GLuint vs = 0, fs = 0, cs = 0, active = 0;  // program names (0 = none); active: glActiveShaderProgram
    bool bound = false;                         // has been bound once (glIsProgramPipeline)
    bool validated = false;
    std::string log;
};

struct Surface;  // EGL surface (egl.cpp)

// Vertex input follows the GL 4.3 / ES 3.1 model (ARB_vertex_attrib_binding): an attribute has a format
// and points at a buffer binding; the binding has the buffer, offset, stride and divisor.
// glVertexAttribPointer is exactly VertexAttribFormat + VertexAttribBinding(i, i) + BindVertexBuffer(i).
struct VertexAttrib {
    bool enabled = false;
    GLint size = 4;
    GLenum type = gl::FLOAT;
    bool normalized = false, integer = false;
    GLuint relativeOffset = 0;
    GLuint binding = 0;      // index into VertexArray::bindings
    GLsizei userStride = 0;  // the stride argument of glVertexAttribPointer (what glGetVertexAttribiv reports)
};

struct VertexBinding {
    GLuint buffer = 0;       // 0: client memory
    uintptr_t offset = 0;    // byte offset into the buffer, or the client pointer
    GLsizei stride = 0;      // distance between elements (0 = every vertex reads the same element)
    GLuint divisor = 0;
};

struct VertexArray {
    GLuint name = 0;
    std::array<VertexAttrib, kMaxVertexAttribs> attribs{};
    std::array<VertexBinding, kMaxVertexBindings> bindings{};
    GLuint elementBuffer = 0;
    VertexArray() {
        for (int i = 0; i < kMaxVertexAttribs; ++i) attribs[size_t(i)].binding = GLuint(i);
    }
};

// Bytes one element of the attribute occupies in its buffer.
uint32_t vertexAttribBytes(const VertexAttrib& a);

// The one program binary format glGetProgramBinary produces ("VPK1").
constexpr GLenum kProgramBinaryFormat = 0x56504B31;

struct Shader {
    GLuint name = 0;
    GLenum type = 0;
    std::string log;
    // Shared, immutable text: glShaderSource makes a new one, glCompileShader's compiledSource is that same object
    // (no second copy), and a linked program keeps the pointer so deleting the shader frees nothing it needs.
    std::shared_ptr<const std::string> source, compiledSource;
    const std::string& sourceText() const {
        static const std::string none;
        return source ? *source : none;
    }
    bool compiled = false, deletePending = false;
    std::shared_ptr<vpglsl::Shader> parsed;
    int attachedTo = 0;
};

struct Program {
    GLuint name = 0;
    GLuint vs = 0, fs = 0, cs = 0;
    bool linked = false, deletePending = false, validated = false;
    std::string log;
    std::string varyingDiag;  // vpglsl::LinkResult::varyingDiag of the last link
    std::map<std::string, int> attribBindings;
    // What glGetProgramBinary serializes (built on demand, never cached): the link's stage sources (shared with
    // the shaders, not copied), attribute bindings and separable flag as they were at the last successful link.
    struct LinkedBinary {
        bool valid = false, separable = false;
        std::shared_ptr<const std::string> sources[3];  // vertex, fragment, compute
        std::map<std::string, int> attribBindings;
    } bin;
    vpglsl::Reflection refl;
    VkShaderModule vsModule = VK_NULL_HANDLE, fsModule = VK_NULL_HANDLE, csModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    std::vector<uint8_t> defaultBlock;  // std140 bytes of binding 0
    bool defaultDirty = true;
    // Uniform locations: location -> (uniform index, element).
    struct Loc {
        int uniform;
        int element;
    };
    bool binaryRetrievableHint = false, separable = false;
    // Transform feedback: glTransformFeedbackVaryings state (taken by the next link) and what the last link made of it.
    std::vector<std::string> xfbVaryings;
    GLenum xfbMode = 0x8C8C;  // GL_INTERLEAVED_ATTRIBS
    GLenum xfbLinkedMode = 0x8C8C;
    // The vertex shader variant that also writes the capture record (see vpglsl::kXfbBinding), its layouts and pipelines.
    VkShaderModule xfbModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout xfbSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout xfbPipeLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, VkPipeline> xfbPipelines;
    std::vector<Loc> locations;
    std::vector<std::vector<GLint>> samplerUnits;  // per reflected uniform (samplers only): unit per element
    std::vector<GLuint> blockBindings;              // per reflection.blocks entry: GL binding point
    std::vector<GLuint> storageBindings;            // per reflection.storageBlocks entry: GL binding point
    std::unordered_map<uint64_t, VkPipeline> pipelines;
    uint64_t serial = 0;  // unique id (pipeline cache keys)
    ~Program();
};

struct SamplerObj {
    GLuint name = 0;
    SamplerParams params;
};

// Queries are coherent rather than GPU-measured: an occlusion query answers "samples passed" when a draw
// was issued while it was active (conservative, as ANY_SAMPLES_PASSED_CONSERVATIVE allows); TIME_ELAPSED
// is the wall-clock span between Begin and End, TIMESTAMP the wall clock; both are available at once.
struct Query {
    GLuint name = 0;
    GLenum target = 0;
    bool active = false;
    uint64_t result = 0;
    uint64_t beginNs = 0, beginDraws = 0, beginXfb = 0;
    uint32_t slot = UINT32_MAX;  // occlusion: its counter in Context::occPool_ (UINT32_MAX = none: coherent)
    uint64_t endSerial = 0;      // occlusion: the submission that holds its end
};

struct Sync {
    uint64_t serial = 0;
};

// Objects shared between contexts of one share group.
struct ShareGroup {
    std::unordered_map<GLuint, std::unique_ptr<Buffer>> buffers;
    std::unordered_map<GLuint, std::unique_ptr<Texture>> textures;
    std::unordered_map<GLuint, std::unique_ptr<Renderbuffer>> renderbuffers;
    std::unordered_map<GLuint, std::unique_ptr<Shader>> shaders;   // shaders and programs share names
    std::unordered_map<GLuint, std::unique_ptr<Program>> programs;
    std::unordered_map<GLuint, std::unique_ptr<SamplerObj>> samplers;
    std::unordered_map<GLsync, std::unique_ptr<Sync>> syncs;
    GLuint nextName = 1, nextShaderName = 1;
    int refs = 0;
    std::map<std::pair<GLenum, GLuint>, std::string> labels;  // glObjectLabel: (identifier, name) -> label
    std::map<GLsync, std::string> syncLabels;
};

// ---- the context -----------------------------------------------------------------------------------
struct BlendState {
    bool enabled = false;
    GLenum srcRGB = gl::ONE, dstRGB = gl::ZERO, srcA = gl::ONE, dstA = gl::ZERO;
    GLenum eqRGB = gl::FUNC_ADD, eqA = gl::FUNC_ADD;
    bool mask[4] = {true, true, true, true};
};

struct StencilFace {
    GLenum func = gl::ALWAYS;
    GLint ref = 0;
    GLuint mask = ~0u, writeMask = ~0u;
    GLenum fail = gl::KEEP, zfail = gl::KEEP, zpass = gl::KEEP;
};

class Context {
public:
    Context(Device* dev, ShareGroup* share, int major, int minor);
    ~Context();

    Device* dev;
    ShareGroup* share;
    int major, minor;
    GLenum error = gl::NO_ERROR;
    // Diagnostics: the first GL_INVALID_ENUM / INVALID_VALUE / INVALID_OPERATION raised from each source
    // line is logged with the vpgles function and line that rejected the call, the enum argument that
    // call had (arg, ~0 when unknown) and the calling thread.
    void setError(GLenum e, GLenum arg = 0xFFFFFFFFu, const char* fn = __builtin_FUNCTION(),
                  unsigned line = __builtin_LINE(), const char* file = __builtin_FILE()) {
        if (e == gl::INVALID_ENUM || e == gl::INVALID_VALUE || e == gl::INVALID_OPERATION) {
            static std::map<std::string, bool> seen;
            static std::mutex seenMu;
            std::lock_guard<std::mutex> g(seenMu);
            if (seen.emplace(std::string(file) + ":" + std::to_string(line), true).second)
                logf("gles.diag %s first at %s:%u (%s) enum=0x%X thread=%zx", e == gl::INVALID_ENUM ? "INVALID_ENUM" : e == gl::INVALID_VALUE ? "INVALID_VALUE" : "INVALID_OPERATION",
                     file, line, fn, arg, std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xFFFFFF);
        }
        if (debugOutput && e != gl::NO_ERROR) debugMessage(gl::DEBUG_SOURCE_API, gl::DEBUG_TYPE_ERROR, e, gl::DEBUG_SEVERITY_HIGH, errorName(e));
        if (error == gl::NO_ERROR) error = e;
    }
    static const char* errorName(GLenum e);

    // Default framebuffer (bound EGL surfaces).
    Surface* draw = nullptr;
    Surface* read = nullptr;

    // EGL: hilo que lo tiene actual (un contexto solo puede ser actual en un hilo) y destrucción
    // diferida (eglDestroyContext sobre un contexto actual lo borra al dejar de serlo). Device::lock.
    bool owned = false, dead = false;
    std::thread::id owner;

    // ---- bindings
    GLuint arrayBuffer = 0, copyRead = 0, copyWrite = 0, pixelPack = 0, pixelUnpack = 0, uniformBuffer = 0;
    // Generic binding points of targets without a use in this layer yet (valid to bind, not consumed):
    // SHADER_STORAGE, ATOMIC_COUNTER, DRAW_INDIRECT, DISPATCH_INDIRECT, TRANSFORM_FEEDBACK, TEXTURE_BUFFER.
    GLuint inertBuffer[6] = {};
    struct IndexedBuffer {
        GLuint buffer = 0;
        GLintptr offset = 0;
        GLsizeiptr size = 0;
    };
    std::array<IndexedBuffer, kMaxUniformBufferBindings> uniformBindings{};
    // Indexed binding points of the targets the compute / transform feedback paths consume.
    std::array<IndexedBuffer, kMaxShaderStorageBindings> ssboBindings{};
    std::array<IndexedBuffer, kMaxAtomicCounterBindings> atomicBindings{};
    // Transform feedback objects (not shared between contexts; 0 is the default object, always there). The indexed buffer
    // bindings, the active / paused state and the capture position belong to the object.
    struct TransformFeedback {
        GLuint name = 0;
        bool bound = false;  // bound once: glIsTransformFeedback
        std::array<IndexedBuffer, kMaxTransformFeedbackBindings> buffers{};
        bool active = false, paused = false;
        GLenum mode = 0;                     // glBeginTransformFeedback's primitiveMode
        uint64_t program = 0;                // Program::serial it began with
        std::array<uint64_t, kMaxTransformFeedbackBindings> used{};  // bytes captured into each binding since Begin
    };
    std::unordered_map<GLuint, std::unique_ptr<TransformFeedback>> transformFeedbacks;
    GLuint transformFeedback = 0, nextTransformFeedback = 1;
    TransformFeedback& curTf();
    uint64_t xfbPrimitivesWritten = 0;  // running count GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN queries difference
    // glBindImageTexture units.
    struct ImageUnit {
        GLuint texture = 0;
        GLint level = 0, layer = 0;
        GLboolean layered = 0;
        GLenum access = gl::READ_ONLY, format = 0;
    };
    std::array<ImageUnit, kMaxImageUnits> images{};
    // Queries: the active query of each target, and the draw counter occlusion queries compare.
    std::map<GLenum, GLuint> activeQuery;
    uint64_t drawCount = 0;
    // Occlusion queries are measured by VPVulkan's visibility counters: one pool per context, a slot per query object.
    VkQueryPool occPool_ = VK_NULL_HANDLE;
    std::vector<bool> occBusy_;
    uint32_t occActive_ = UINT32_MAX;  // the slot counting draws now
    uint32_t occSlot();                // a free slot (UINT32_MAX: none)
    void occBegin(Query& q);
    void occEnd(Query& q);
    bool occResult(Query& q, bool wait, uint64_t& value);  // false: not available
    void occPause();                   // internal draws (clears) do not count
    void occResume();
    // KHR_debug.
    using DebugProc = void (*)(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*, const void*);
    DebugProc debugCb = nullptr;
    const void* debugUser = nullptr;
    bool debugOutput = false, debugSync = false, inDebugCb = false;
    struct DebugMessage {
        GLenum source, type, id, severity;
        std::string text;
    };
    std::vector<DebugMessage> debugLog;
    struct DebugRule {
        GLenum source, type, severity;
        std::vector<GLuint> ids;
        bool enabled;
    };
    std::vector<DebugRule> debugRules;
    std::vector<std::string> debugGroups;  // pushed groups (the implicit top level is not stored)
    void debugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, const std::string& text);
    GLuint drawFbo = 0, readFbo = 0, renderbuffer = 0, program = 0, vao = 0;
    std::unordered_map<GLuint, std::unique_ptr<ProgramPipeline>> programPipelines;
    GLuint programPipeline = 0, nextProgramPipeline = 1;
    // glSampleMaski: stored only (rendering is 1x, the coverage mask has nothing to apply to).
    GLbitfield sampleMask = ~0u;
    bool sampleMaskWarned = false;
    int activeUnit = 0;
    struct Unit {
        GLuint tex2D = 0, texCube = 0, tex3D = 0, tex2DArray = 0, texExternal = 0, tex2DMS = 0, sampler = 0;
        // Targets that only bind / take parameters here (no sampling path): kept so engines probing them
        // get a valid binding instead of INVALID_ENUM.
        GLuint texCubeArray = 0, tex2DMSArray = 0, texBuffer = 0;
    };
    std::array<Unit, kMaxTextureUnits> units{};
    std::unordered_map<GLuint, std::unique_ptr<Framebuffer>> fbos;
    std::unordered_map<GLuint, std::unique_ptr<VertexArray>> vaos;  // 0 = default VAO
    std::unordered_map<GLuint, std::unique_ptr<Query>> queries;
    GLuint nextFbo = 1, nextVao = 1, nextQuery = 1;
    std::array<std::array<float, 4>, kMaxVertexAttribs> currentAttrib{};
    std::array<uint8_t, kMaxVertexAttribs> currentAttribKind{};  // 0 float, 1 int, 2 uint

    // ---- fixed-function state
    GLint viewport[4] = {0, 0, 0, 0};
    GLint scissor[4] = {0, 0, 0, 0};
    float depthRange[2] = {0.0f, 1.0f};
    bool scissorTest = false, depthTest = false, depthMask = true, stencilTest = false, cullFace = false,
         polygonOffsetFill = false, sampleAlphaToCoverage = false, sampleCoverage = false, primitiveRestart = false,
         rasterizerDiscard = false, dither = true, blendEnabledAll = false;
    GLenum depthFunc = gl::LESS, cullMode = gl::BACK, frontFace = gl::CCW;
    std::array<BlendState, kMaxDrawBuffers> blend{};
    float blendColor[4] = {0, 0, 0, 0};
    StencilFace stencil[2];  // front, back
    float polygonOffset[2] = {0, 0};  // factor, units
    float lineWidth = 1.0f;
    float clearColor[4] = {0, 0, 0, 0};
    float clearDepth = 1.0f;
    GLint clearStencil = 0;
    GLint unpackAlignment = 4, packAlignment = 4, unpackRowLength = 0, unpackSkipRows = 0, unpackSkipPixels = 0,
          unpackImageHeight = 0, unpackSkipImages = 0, packRowLength = 0, packSkipRows = 0, packSkipPixels = 0;
    GLenum hints[2] = {gl::DONT_CARE, gl::DONT_CARE};
    float sampleCoverageValue = 1.0f;
    bool sampleCoverageInvert = false;

    // ---- object helpers
    Buffer* buffer(GLuint n) const;
    Texture* texture(GLuint n) const;
    Renderbuffer* rbo(GLuint n) const;
    Program* prog(GLuint n) const;
    // The program draws / dispatches / glUniform* use: the current program, else the bound pipeline's.
    Program* graphicsProgram();
    Program* computeProgram();
    Program* uniformProgram();
    Shader* shader(GLuint n) const;
    Framebuffer* fbo(GLuint n) const;
    VertexArray* currentVao();
    GLuint* bufferBinding(GLenum target);
    GLuint* textureBinding(GLenum target, int unit);
    Texture* boundTexture(GLenum target);
    GLuint genName();

    // ---- recording
    VkCommandBuffer cmd();             // the open command buffer (begun on demand)
    void endPass();                    // closes an open dynamic rendering pass
    void barrier();                    // full memory barrier (outside passes)
    uint64_t flush(bool wait);         // submits; returns the submission serial (0 if nothing)
    void keep(std::shared_ptr<void> p) { keep_.push_back(std::move(p)); }
    void useStorage(const StorageP& s);  // marks it used by the open command buffer
    // image2D bound to one slice of a 3D texture: Metal cannot view a 3D texture as 2D, so the dispatch works on a 2D
    // copy of the slice (copied in before, back after: syncSlicesBack).
    struct SliceSync {
        ImageP proxy, source;
        uint32_t level = 0, slice = 0;
        bool writeBack = false;
    };
    std::vector<SliceSync> sliceBack_;
    std::map<std::pair<const Image*, uint64_t>, ImageP> sliceProxies_;
    void syncSlicesBack();
    // Transient upload memory for this submission (vertex/index streaming, uniforms, staging).
    struct Alloc {
        VkBuffer buffer;
        VkDeviceSize offset;
        uint8_t* host;
    };
    Alloc stream(VkDeviceSize size, VkDeviceSize align = 16);
    VkDescriptorSet allocSet(VkDescriptorSetLayout layout);

    // Copy-on-write before a CPU write to a buffer the GPU may use.
    void prepareWrite(Buffer& b, bool discard);

    // Images
    bool realizeTexture(Texture& t);  // (re)creates t.image from its level definitions
    // Gives t an image with STORAGE_BIT (glBindImageTexture), keeping the contents. False when it cannot be image-bound.
    bool makeStorage(Texture& t);
    void imageInit(Image& img);       // UNDEFINED -> GENERAL
    VkSampler samplerFor(const SamplerParams& p, const Texture& t);

    // Framebuffer resolution: the attachments of the bound draw (or read) framebuffer.
    struct FbTarget {
        Image* image = nullptr;
        uint32_t level = 0, layer = 0, layers = 1;
        bool multiview = false;
    };
    struct FbInfo {
        std::array<FbTarget, kMaxDrawBuffers> color{};
        FbTarget depth, stencil;
        uint32_t width = 0, height = 0, viewCount = 0;
        GLenum status = gl::FRAMEBUFFER_COMPLETE;
        bool defaultFb = false;
    };
    FbInfo resolveFramebuffer(GLuint fbo, bool forDraw);
    bool targetOf(const Attachment& a, FbTarget& out, uint32_t& w, uint32_t& h);

    // Pass management
    // Opens (or continues) a pass on `fb`. `firstClear`: full-target clears that become the new pass's CLEAR load ops
    // (emptied when it takes them).
    bool beginPass(const FbInfo& fb, std::vector<VkClearAttachment>* firstClear = nullptr);
    bool inPass() const { return passOpen_; }
    // glInvalidateFramebuffer / glInvalidateSubFramebuffer (full rect) / glDiscardFramebufferEXT.
    void invalidateFramebuffer(GLenum target, GLsizei n, const GLenum* attachments);

    // Draw / clear
    void drawArrays(GLenum mode, GLint first, GLsizei count, GLsizei instances);
    void drawElements(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLint baseVertex);
    void clear(GLbitfield mask);
    void clearBuffer(GLenum buffer, GLint drawbuffer, const void* value, GLenum kind, float depth, GLint stencil);
    void readPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels);
    void blitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1,
                         GLbitfield mask, GLenum filter);
    void dispatch(GLuint x, GLuint y, GLuint z);
    void dispatchIndirect(GLintptr offset);
    // glDrawArraysIndirect (indexType 0) / glDrawElementsIndirect: the command comes from the bound
    // DRAW_INDIRECT_BUFFER at byte offset `indirect`.
    void drawIndirect(GLenum mode, GLenum indexType, const void* indirect);
    void copyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint sx, GLint sy, GLint sz, GLuint dstName,
                          GLenum dstTarget, GLint dstLevel, GLint dx, GLint dy, GLint dz, GLsizei w, GLsizei h, GLsizei d);
    // The VkBufferView of a buffer texture (created on demand), or null when it has no usable buffer.
    VkBufferView texelBufferView(Texture& t);
    VkBufferView blackTexelBufferView();

    // Texture transfers
    void texImage(GLenum target, GLint level, GLenum internalFormat, GLsizei w, GLsizei h, GLsizei d, GLenum format,
                  GLenum type, const void* pixels, bool sub, GLint xo, GLint yo, GLint zo, bool compressed,
                  GLsizei imageSize);
    void texStorage(GLenum target, GLsizei levels, GLenum internalFormat, GLsizei w, GLsizei h, GLsizei d,
                    GLsizei samples);
    void copyTexImage(GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei w, GLsizei h,
                      bool sub, GLint xo, GLint yo, GLint zo);
    void generateMipmap(GLenum target);
    void copyBufferToImage(Image& img, uint32_t level, uint32_t layer, int32_t x, int32_t y, int32_t z, uint32_t w,
                           uint32_t h, uint32_t d, const uint8_t* data, size_t bytes);
    // Keeps an upload to a level above 0 until level 0 defines the image (Texture::pending) / uploads the kept ones.
    void queuePending(Texture& t, uint32_t level, uint32_t layer, int32_t x, int32_t y, int32_t z, uint32_t w, uint32_t h,
                      uint32_t d, const uint8_t* data, size_t bytes);
    void flushPending(Texture& t);

    // Program use
    enum class Kind { Float, Int, Uint, Bool };
    bool linkProgram(Program& p);
    // The link of linkProgram on explicit shaders (glProgramBinary re-links unregistered ones).
    bool linkShaders(Program& p, Shader* vs, Shader* fs, Shader* cs);
    // glProgramBinary: false (link status false, log set) for a foreign / corrupt blob; never a GL error.
    bool loadProgramBinary(Program& p, GLenum format, const void* data, size_t bytes);
    // The glGetProgramBinary blob of `p`'s last successful link, built now (empty = no linked program).
    static std::vector<uint8_t> programBinary(const Program& p);
    // glUniform*: `cols`/`rows` > 1 for matrices (data column-major unless transpose).
    // `target` = the program to write (glProgramUniform*), null = the current program.
    void setUniform(GLint location, int components, Kind kind, GLsizei count, const void* v, int cols, bool transpose,
                    Program* target = nullptr);
    void getUniform(Program& p, GLint location, void* out, Kind kind);

private:
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<void>> keep_;
    std::vector<StorageP> used_;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    bool passOpen_ = false;
    // Attachments of the open pass invalidated since it began (bit per slot: colour attachment i, kMaxDrawBuffers =
    // depth, +1 = stencil). glInvalidateFramebuffer comes after the pass began, so VPVulkan is told (vpvk_cmd_pass_drop)
    // to store them DONT_CARE; a later write takes the bit back.
    uint32_t passDrop_ = 0;
    void setPassDrop(uint32_t mask);
    struct PassKey {
        std::array<VkImageView, kMaxDrawBuffers> color{};
        VkImageView depth = VK_NULL_HANDLE, stencil = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0, viewMask = 0;
        bool operator==(const PassKey& o) const;
    } pass_;
    PassKey passKeyFor(const FbInfo& fb);  // the attachments a pass on `fb` would have
    // Streaming ring (per submission chunks).
    struct Chunk {
        StorageP storage;
        VkDeviceSize used = 0;
    };
    std::vector<Chunk> chunks_;
    std::vector<StorageP> spareChunks_;
    std::vector<std::pair<SamplerParams, VkSampler>> samplerCache_;
    std::vector<bool> samplerCacheMip_;
    std::vector<VkSampler> samplerCacheList_;

    bool prepareDraw(GLenum mode, VkPrimitiveTopology& topo, uint32_t vertexCount, uint32_t firstVertex,
                     uint32_t instances, bool indexed, VkBuffer& vbufs0);
    VkPipeline pipelineFor(Program& p, const FbInfo& fb, VkPrimitiveTopology topo, const std::vector<VkVertexInputBindingDescription>& binds,
                           const std::vector<VkVertexInputAttributeDescription>& attrs, bool xfb = false);
    bool bindResources(Program& p, const StorageP& xfbScratch = nullptr);
    // Transform feedback while a draw is recorded: the capture pass (vertex shader variant writing the scratch buffer with the
    // rasterizer off), then the captured vertices are put in primitive order into the bound buffers (see captureTransformFeedback).
    bool captureTransformFeedback(GLenum mode, uint32_t count, uint32_t first, uint32_t instances, bool indexed,
                                  const std::vector<uint32_t>& indices, int32_t baseVertex);
    StorageP xfbScratch_;  // set while the capture draw is prepared
    bool beginDispatch(Program*& p);
    bool dispatching_ = false;  // bindResources runs for a dispatch (copies may be recorded)
    // The 2D storage view for image2D bound to slice `layer` of a 3D texture (see SliceSync); null when it cannot be had.
    VkImageView sliceView(Texture& t, uint32_t level, GLint layer, GLenum access, VkFormat format);
    bool attribsDirect(Program& p, VertexArray* va);
    // glClear / glClearBuffer with a colour or stencil write mask: vkCmdClearAttachments cannot mask channels or bits, so
    // those are a full-target triangle drawn with the mask as the pipeline's write mask (what ANGLE does). `forcedProgram_`
    // is the program graphicsProgram() answers with while that draw is recorded.
    struct ClearDraw {
        uint32_t colorBits = 0;                                        // draw buffers to clear
        std::array<std::array<uint32_t, 4>, kMaxDrawBuffers> value{};  // their clear values (float / int / uint bits)
        bool stencil = false;
        uint32_t stencilValue = 0;
    };
    Program* forcedProgram_ = nullptr;
    std::map<std::string, std::unique_ptr<Program>> clearProgs_;  // per colour output layout ("f" / "i" / "u" per slot)
    Program* clearProgram(const std::string& kinds);
    bool clearByDraw(const FbInfo& fb, const VkRect2D& rect, const ClearDraw& cd);
    StorageP blackTexelStorage_;
    std::shared_ptr<TexelBufferView> blackTexelView_;
    void setDynamicState(const FbInfo& fb);
    FbInfo drawFb_;
    uint32_t indexMax_ = 0;
    bool provokeOk_ = false;  // set by drawArrays/drawElements, the draws that can run as a rotated list; read once by prepareDraw
    bool provoke_ = false;  // this draw runs as an index list with each primitive's last vertex first (flat varyings, see prepareDraw)
    std::unique_ptr<Texture> black_;
    Texture& blackTexture();
    Texture* defaultTexture(GLenum target);  // texture object 0 of the target if the app ever used it, else null
};

GLint uniformLocation(const Program& p, const char* name);

Context* current();
void setCurrent(Context* c);
Context* currentOrLast();
void forgetContext(Context* c);

}  // namespace vpgles
