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
#include "glsl.h"
#include "vk.h"

namespace vpgles {

void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

constexpr int kMaxTextureUnits = 32;
constexpr int kMaxVertexAttribs = 16;
constexpr int kMaxDrawBuffers = 8;
constexpr int kMaxUniformBufferBindings = 36;

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
    std::map<uint64_t, VkImageView> views;  // cached views
    ~Image();
    VkImageAspectFlags aspect() const;
    // A 2D (array) view of one level and `layerCount` layers from `layer`, for attachments.
    VkImageView attachmentView(uint32_t level, uint32_t layer, uint32_t layerCount);
};
using ImageP = std::shared_ptr<Image>;

struct Texture {
    GLuint name = 0;
    GLenum target = 0;  // 0 until first bound
    ImageP image;
    bool immutable = false;
    GLint immutableLevels = 0;
    // Level definitions (glTexImage*): per level, the size and format the app defined.
    struct Level {
        uint32_t w = 0, h = 0, d = 0;
        const FormatInfo* format = nullptr;
    };
    std::vector<Level> levels;
    SamplerParams sampler;
    GLint baseLevel = 0, maxLevel = 1000;
    GLenum swizzle[4] = {gl::RED, gl::GREEN, gl::BLUE, gl::ALPHA};
    GLenum depthStencilMode = gl::DEPTH_COMPONENT;
    bool srgbDecode = true;
    // External OES image (EGLImage / video): sampled from this image.
    VkImageView sampledView();
    uint64_t viewKey() const;
};

struct Renderbuffer {
    GLuint name = 0;
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
    std::array<Attachment, kMaxDrawBuffers> color{};
    Attachment depth, stencil;
    std::array<GLenum, kMaxDrawBuffers> drawBuffers{};
    GLenum readBuffer = gl::COLOR_ATTACHMENT0;
    Framebuffer() {
        drawBuffers.fill(gl::NONE);
        drawBuffers[0] = gl::COLOR_ATTACHMENT0;
    }
};

struct Surface;  // EGL surface (egl.cpp)

struct VertexAttrib {
    bool enabled = false;
    GLint size = 4;
    GLenum type = gl::FLOAT;
    bool normalized = false, integer = false;
    GLsizei stride = 0;
    GLuint buffer = 0;          // 0: client memory
    const void* pointer = nullptr;  // offset into the buffer, or a client pointer
    GLuint divisor = 0;
};

struct VertexArray {
    GLuint name = 0;
    std::array<VertexAttrib, kMaxVertexAttribs> attribs{};
    GLuint elementBuffer = 0;
};

struct Shader {
    GLuint name = 0;
    GLenum type = 0;
    std::string source, log;
    bool compiled = false, deletePending = false;
    std::shared_ptr<vpglsl::Shader> parsed;
    int attachedTo = 0;
};

struct Program {
    GLuint name = 0;
    GLuint vs = 0, fs = 0, cs = 0;
    bool linked = false, deletePending = false, validated = false;
    std::string log;
    std::map<std::string, int> attribBindings;
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
    std::vector<Loc> locations;
    std::vector<std::vector<GLint>> samplerUnits;  // per reflected uniform (samplers only): unit per element
    std::vector<GLuint> blockBindings;              // per reflection.blocks entry: GL binding point
    std::unordered_map<uint64_t, VkPipeline> pipelines;
    uint64_t serial = 0;  // unique id (pipeline cache keys)
    ~Program();
};

struct SamplerObj {
    GLuint name = 0;
    SamplerParams params;
};

struct Query {
    GLuint name = 0;
    GLenum target = 0;
    bool active = false;
    uint64_t result = 1;
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
    void setError(GLenum e) {
        if (error == gl::NO_ERROR) error = e;
    }

    // Default framebuffer (bound EGL surfaces).
    Surface* draw = nullptr;
    Surface* read = nullptr;

    // ---- bindings
    GLuint arrayBuffer = 0, copyRead = 0, copyWrite = 0, pixelPack = 0, pixelUnpack = 0, uniformBuffer = 0;
    struct IndexedBuffer {
        GLuint buffer = 0;
        GLintptr offset = 0;
        GLsizeiptr size = 0;
    };
    std::array<IndexedBuffer, kMaxUniformBufferBindings> uniformBindings{};
    GLuint drawFbo = 0, readFbo = 0, renderbuffer = 0, program = 0, vao = 0;
    int activeUnit = 0;
    struct Unit {
        GLuint tex2D = 0, texCube = 0, tex3D = 0, tex2DArray = 0, texExternal = 0, tex2DMS = 0, sampler = 0;
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
    bool beginPass(const FbInfo& fb);  // opens (or continues) a pass on `fb`
    bool inPass() const { return passOpen_; }

    // Draw / clear
    void drawArrays(GLenum mode, GLint first, GLsizei count, GLsizei instances);
    void drawElements(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLint baseVertex);
    void clear(GLbitfield mask);
    void clearBuffer(GLenum buffer, GLint drawbuffer, const void* value, GLenum kind, float depth, GLint stencil);
    void readPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels);
    void blitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1,
                         GLbitfield mask, GLenum filter);
    void dispatch(GLuint x, GLuint y, GLuint z);

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

    // Program use
    enum class Kind { Float, Int, Uint, Bool };
    bool linkProgram(Program& p);
    // glUniform*: `cols`/`rows` > 1 for matrices (data column-major unless transpose).
    void setUniform(GLint location, int components, Kind kind, GLsizei count, const void* v, int cols, bool transpose);
    void getUniform(Program& p, GLint location, void* out, Kind kind);

private:
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<void>> keep_;
    std::vector<StorageP> used_;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    bool passOpen_ = false;
    struct PassKey {
        std::array<VkImageView, kMaxDrawBuffers> color{};
        VkImageView depth = VK_NULL_HANDLE, stencil = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0, viewMask = 0;
        bool operator==(const PassKey& o) const;
    } pass_;
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
                           const std::vector<VkVertexInputAttributeDescription>& attrs);
    bool bindResources(Program& p);
    void setDynamicState(const FbInfo& fb);
    FbInfo drawFb_;
    uint32_t indexMax_ = 0;
    std::unique_ptr<Texture> black_;
    Texture& blackTexture();
};

GLint uniformLocation(const Program& p, const char* name);

Context* current();
void setCurrent(Context* c);
Context* currentOrLast();
void forgetContext(Context* c);

}  // namespace vpgles
