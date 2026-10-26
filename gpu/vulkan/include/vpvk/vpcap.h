// VPCAP: the capture file of VPVulkan's trimmed capture (gpu/vulkan/src/capture.cpp) and its reader.
// Phase 2 (vpreplay) reads files through this header and vpcap_calls.h (the typed decoder); nothing here depends on
// the driver. One .vpcap file:
//
//   FileHeader (32 bytes)
//   Block*     { BlockHeader (16 bytes), payload }   until EOF
//
// Blocks (Section): META and END are "key=value\n" text; STATE holds the creation of every live object at capture start
// (+ descriptor contents), FRAME the calls of the captured frames. STATE / FRAME payloads are Call records back to back:
//   CallHeader (12 bytes) { u16 id, u16 flags (0), u32 thread, u32 size }, then `size` payload bytes.
// Each block's payload is stored raw (codec 0) or zlib-deflated (codec 1: zlib "compress2" format; zstd is not in
// third_party/ so the system zlib is used). Little endian, LP64: handles inside payloads are stable ids (0 = null),
// never addresses; Vulkan structs are stored as their raw LP64 bytes with handles replaced by ids and every pointer
// replaced by out-of-line data (see vpcap_calls.h). A frame ends with a vpcapFrameEnd call.
//
// Versioning: kVersion changes when an existing record layout changes; new call ids may appear without a bump
// (a reader skips ids it does not know: size says where the next record starts).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vpcap {

constexpr char kMagic[8] = {'V', 'P', 'C', 'A', 'P', 0, 0, 0};
constexpr uint32_t kVersion = 1;

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t headerBytes;  // sizeof(FileHeader)
  uint32_t flags;        // bit 0: little endian LP64 payloads (always set)
  uint32_t reserved0;
  uint64_t reserved1;
};
static_assert(sizeof(FileHeader) == 32);

enum class Section : uint32_t { Meta = 1, State = 2, Frame = 3, End = 4 };
enum : uint32_t { kCodecRaw = 0, kCodecZlib = 1 };

struct BlockHeader {
  uint32_t section;
  uint32_t codec;
  uint32_t storedBytes;  // payload bytes in the file
  uint32_t rawBytes;     // after decompression
};
static_assert(sizeof(BlockHeader) == 16);

struct CallHeader {
  uint16_t id;
  uint16_t flags;
  uint32_t thread;
  uint32_t size;
};
static_assert(sizeof(CallHeader) == 12);

// Stable call ids (never renumbered). 0x00xx: capture pseudo-calls, 0x01xx: objects / queue, 0x02xx: vkCmd*.
#define VPCAP_PSEUDO(X)                                                                                                \
  X(vpcapFrameEnd, 0x0001)           /* u64 frame index (0-based), u64 vkQueueSubmit count since the capture began */   \
  X(vpcapMemoryData, 0x0002)         /* u64 memory id, u64 offset, u64 size, size bytes: write before the next submit */ \
  X(vpcapImageData, 0x0003)          /* image contents of one subresource (or a byte range of it): see ImageData below */
#define VPCAP_CALLS(X) VPCAP_PSEUDO(X) VPCAP_VKCALLS(X)
#define VPCAP_VKCALLS(X)                                                                                               \
  X(vkCreateInstance, 0x0101)        X(vkDestroyInstance, 0x0102)       X(vkCreateDevice, 0x0103)          X(vkDestroyDevice, 0x0104)                       \
  X(vkGetDeviceQueue, 0x0105)                                                                                            \
  X(vkAllocateMemory, 0x0110)        X(vkFreeMemory, 0x0111)            X(vkMapMemory, 0x0112)                           \
  X(vkUnmapMemory, 0x0113)           X(vkFlushMappedMemoryRanges, 0x0114)                                                \
  X(vkCreateBuffer, 0x0120)          X(vkDestroyBuffer, 0x0121)         X(vkBindBufferMemory, 0x0122)                    \
  X(vkCreateBufferView, 0x0123)      X(vkDestroyBufferView, 0x0124)                                                      \
  X(vkCreateImage, 0x0130)           X(vkDestroyImage, 0x0131)          X(vkBindImageMemory, 0x0132)                     \
  X(vkCreateImageView, 0x0133)       X(vkDestroyImageView, 0x0134)                                                       \
  X(vkCreateSampler, 0x0140)         X(vkDestroySampler, 0x0141)                                                         \
  X(vkCreateDescriptorSetLayout, 0x0150) X(vkDestroyDescriptorSetLayout, 0x0151)                                         \
  X(vkCreatePipelineLayout, 0x0152)  X(vkDestroyPipelineLayout, 0x0153)                                                  \
  X(vkCreateDescriptorPool, 0x0154)  X(vkDestroyDescriptorPool, 0x0155) X(vkResetDescriptorPool, 0x0156)                 \
  X(vkAllocateDescriptorSets, 0x0157) X(vkFreeDescriptorSets, 0x0158)   X(vkUpdateDescriptorSets, 0x0159)               \
  X(vkCreateRenderPass, 0x0160)      X(vkCreateRenderPass2, 0x0161)     X(vkDestroyRenderPass, 0x0162)                   \
  X(vkCreateFramebuffer, 0x0163)     X(vkDestroyFramebuffer, 0x0164)                                                     \
  X(vkCreateShaderModule, 0x0170)    X(vkDestroyShaderModule, 0x0171)                                                    \
  X(vkCreateGraphicsPipelines, 0x0172) X(vkCreateComputePipelines, 0x0173) X(vkDestroyPipeline, 0x0174)                  \
  X(vkCreateCommandPool, 0x0180)     X(vkDestroyCommandPool, 0x0181)    X(vkResetCommandPool, 0x0182)                    \
  X(vkAllocateCommandBuffers, 0x0183) X(vkFreeCommandBuffers, 0x0184)   X(vkBeginCommandBuffer, 0x0185)                  \
  X(vkEndCommandBuffer, 0x0186)      X(vkResetCommandBuffer, 0x0187)                                                     \
  X(vkCreateFence, 0x0190)           X(vkDestroyFence, 0x0191)          X(vkResetFences, 0x0192)                         \
  X(vkWaitForFences, 0x0193)         X(vkCreateSemaphore, 0x0194)       X(vkDestroySemaphore, 0x0195)                    \
  X(vkCreateEvent, 0x0196)           X(vkDestroyEvent, 0x0197)          X(vkQueueSubmit, 0x0198)                         \
  X(vkQueueWaitIdle, 0x0199)         X(vkDeviceWaitIdle, 0x019A)        X(vkQueueSubmit2, 0x019B)                        \
  X(vkCreateQueryPool, 0x01A0)       X(vkDestroyQueryPool, 0x01A1)                                                       \
  X(vkCmdBeginRenderPass, 0x0200)    X(vkCmdBeginRenderPass2, 0x0201)   X(vkCmdEndRenderPass, 0x0202)                   \
  X(vkCmdEndRenderPass2, 0x0203)     X(vkCmdNextSubpass, 0x0204)        X(vkCmdBeginRendering, 0x0205)                   \
  X(vkCmdEndRendering, 0x0206)       X(vkCmdBindPipeline, 0x0207)       X(vkCmdBindDescriptorSets, 0x0208)               \
  X(vkCmdPushConstants, 0x0209)      X(vkCmdBindVertexBuffers, 0x020A)  X(vkCmdBindVertexBuffers2, 0x020B)               \
  X(vkCmdBindIndexBuffer, 0x020C)    X(vkCmdSetViewport, 0x020D)        X(vkCmdSetScissor, 0x020E)                       \
  X(vkCmdSetViewportWithCount, 0x020F) X(vkCmdSetScissorWithCount, 0x0210) X(vkCmdSetStencilReference, 0x0211)          \
  X(vkCmdSetBlendConstants, 0x0212)  X(vkCmdSetLineWidth, 0x0213)       X(vkCmdSetDepthBias, 0x0214)                     \
  X(vkCmdSetDepthBounds, 0x0215)     X(vkCmdSetStencilCompareMask, 0x0216) X(vkCmdSetStencilWriteMask, 0x0217)           \
  X(vkCmdSetCullMode, 0x0218)        X(vkCmdSetFrontFace, 0x0219)       X(vkCmdSetPrimitiveTopology, 0x021A)             \
  X(vkCmdSetDepthTestEnable, 0x021B) X(vkCmdSetDepthWriteEnable, 0x021C) X(vkCmdSetDepthCompareOp, 0x021D)               \
  X(vkCmdSetDepthBoundsTestEnable, 0x021E) X(vkCmdSetStencilTestEnable, 0x021F) X(vkCmdSetStencilOp, 0x0220)             \
  X(vkCmdSetRasterizerDiscardEnable, 0x0221) X(vkCmdSetDepthBiasEnable, 0x0222) X(vkCmdSetPrimitiveRestartEnable, 0x0223) \
  X(vkCmdDraw, 0x0230)               X(vkCmdDrawIndexed, 0x0231)        X(vkCmdDrawIndirect, 0x0232)                     \
  X(vkCmdDrawIndexedIndirect, 0x0233) X(vkCmdDispatch, 0x0234)          X(vkCmdDispatchIndirect, 0x0235)                 \
  X(vkCmdDrawIndirectCount, 0x0236)  X(vkCmdDrawIndexedIndirectCount, 0x0237)                                           \
  X(vkCmdCopyBuffer, 0x0240)         X(vkCmdCopyImage, 0x0241)          X(vkCmdCopyBufferToImage, 0x0242)                \
  X(vkCmdCopyImageToBuffer, 0x0243)  X(vkCmdBlitImage, 0x0244)          X(vkCmdResolveImage, 0x0245)                     \
  X(vkCmdClearColorImage, 0x0246)    X(vkCmdClearDepthStencilImage, 0x0247) X(vkCmdClearAttachments, 0x0248)             \
  X(vkCmdFillBuffer, 0x0249)         X(vkCmdUpdateBuffer, 0x024A)       X(vkCmdPipelineBarrier, 0x024B)                  \
  X(vkCmdExecuteCommands, 0x024C)                                                                                        \
  X(vkCmdCopyBuffer2, 0x0250)        X(vkCmdCopyImage2, 0x0251)         X(vkCmdCopyBufferToImage2, 0x0252)               \
  X(vkCmdCopyImageToBuffer2, 0x0253) X(vkCmdBlitImage2, 0x0254)         X(vkCmdResolveImage2, 0x0255)                    \
  X(vkCmdPipelineBarrier2, 0x0256)                                                                                       \
  X(vkCmdSetEvent, 0x0260)           X(vkCmdSetEvent2, 0x0261)          X(vkCmdResetEvent, 0x0262)                       \
  X(vkCmdResetEvent2, 0x0263)        X(vkCmdWaitEvents, 0x0264)         X(vkCmdWaitEvents2, 0x0265)                      \
  X(vkCmdBeginQuery, 0x0270)         X(vkCmdEndQuery, 0x0271)           X(vkCmdResetQueryPool, 0x0272)                   \
  X(vkCmdWriteTimestamp, 0x0273)     X(vkCmdWriteTimestamp2, 0x0274)

namespace id {
#define VPCAP_X(name, num) constexpr uint16_t name = num;
VPCAP_CALLS(VPCAP_X)
#undef VPCAP_X
}  // namespace id

// "vkCmdDraw"; "unknown" for an id this build does not list.
const char* callName(uint16_t callId);

// Writes a .vpcap file. Not thread-safe (the capture serialises it).
class Writer {
 public:
  ~Writer();
  bool open(const std::string& path);
  void meta(const std::string& text);              // META block (key=value lines), written at once
  void begin(Section s);                           // later calls land in a block of this section
  void append(const void* record, size_t bytes);   // one complete record (CallHeader + payload)
  bool close(const std::string& endText);          // flushes, writes END, closes; false on any I/O error
  void sync();                                     // writes the open block and flushes the file (a capture cut short stays readable)
  bool ok() const { return ok_; }
  uint64_t fileBytes() const { return fileBytes_; }  // bytes written so far (the open block excluded)
  uint64_t rawBytes() const { return rawBytes_; }    // uncompressed payload bytes appended so far
  uint64_t calls() const { return calls_; }

 private:
  void flush();
  void writeBlock(Section s, const uint8_t* data, size_t n);
  std::FILE* f_ = nullptr;
  Section cur_ = Section::State;
  std::vector<uint8_t> buf_;
  bool ok_ = true;
  uint64_t fileBytes_ = 0, rawBytes_ = 0, calls_ = 0;
};

// One call record as stored. `data` points into the reader's block buffer: valid until the next next().
struct Call {
  uint16_t id = 0;
  uint32_t thread = 0;
  const uint8_t* data = nullptr;
  uint32_t size = 0;
  Section section = Section::State;
  uint32_t frame = 0;  // Frame section: 0-based index of the frame the call belongs to (vpcapFrameEnd closes it)
  uint64_t index = 0;  // position among all records of the file
};

class Reader {
 public:
  ~Reader();
  bool open(const std::string& path, std::string* error = nullptr);
  uint32_t version() const { return version_; }
  const std::string& meta() const { return meta_; }  // META text; END text is appended when the end is reached
  bool next(Call& out);                              // false at the end of file (or on a corrupt block: error())
  const std::string& error() const { return error_; }
  // After next() returned false: the END block was seen and no error occurred. False: a capture cut short (the writer syncs after
  // every frame, so such a file is readable up to its last frame) or a damaged file.
  bool complete() const { return ended_ && error_.empty(); }

 private:
  bool loadBlock();
  std::FILE* f_ = nullptr;
  uint32_t version_ = 0;
  std::string meta_, error_;
  std::vector<uint8_t> block_;
  size_t pos_ = 0;
  Section section_ = Section::State;
  uint32_t frame_ = 0;
  uint64_t index_ = 0;
  bool done_ = false, ended_ = false;
  long fileBytes_ = -1;  // size of the file (-1: unknown, e.g. a pipe)
};

// vpcapMemoryData payload.
struct MemoryData {
  uint64_t memory = 0, offset = 0, size = 0;
  const uint8_t* data = nullptr;
};
bool decodeMemoryData(const Call& c, MemoryData& out);
// vpcapImageData payload (written in the FRAME section right before the vkQueueSubmit that first reads the image, like
// vpcapMemoryData): the contents of subresource (mip, layer, aspect) of image `image` as the GPU holds them, in the
// native layout of the texture backend, tightly packed: depth slices of rows of blocks (a compressed format as stored
// block by block, partial blocks at the edges included). width / height / depth are the texels of that mip level. The
// subresource may be split in several records: `offset` is the byte position of `data` in the packed subresource, `total`
// the size of all of it. A subresource that is all zero is not written (a new image reads as zero). Layout:
//   u64 image id, u32 mip, u32 layer, u32 aspect (VkImageAspectFlagBits: COLOR 1, DEPTH 2, STENCIL 4; a multi-planar image: PLANE_0 0x10,
//   PLANE_1 0x20, PLANE_2 0x40), u32 width, u32 height, u32 depth, u32 store, u32 reserved (0), u64 total, u64 offset, u64 size, size bytes.
// `store`: how the bytes relate to the image's Vulkan format. 0: the Vulkan layout (a format the backend holds under another
// layout is not written: D24 depth, counted as unsupported); 1: B4G4R4A4 held as ABGR4 (the R and B nibbles of each texel swapped).
enum : uint32_t { kStoreVulkan = 0, kStoreSwappedRB4 = 1 };
constexpr size_t kImageDataHeader = 64;
struct ImageData {
  uint64_t image = 0;
  uint32_t mip = 0, layer = 0, aspect = 0, width = 0, height = 0, depth = 0, store = 0;
  uint64_t total = 0, offset = 0, size = 0;
  const uint8_t* data = nullptr;
};
bool decodeImageData(const Call& c, ImageData& out);
// vpcapFrameEnd payload.
bool decodeFrameEnd(const Call& c, uint64_t& frame, uint64_t& submits);

}  // namespace vpcap

#ifdef __cplusplus
extern "C" {
#endif
// Trimmed capture of VPVulkan (gpu/vulkan/src/capture.cpp). Look up by name (dlsym) like the vpvk_config_* functions.
// Tracking (the retained creation of every live object) must be on from before the objects are created: environment
// VPVK_CAPTURE=1 (or VPVK_CAPTURE_FRAME, which implies it) or vpvk_capture_track(1) before vkCreateInstance. The cost
// is the memory of the create infos and SPIR-V of the live objects, plus one atomic load per hooked call.
//
// Request: capture `frames` (>= 1) frames beginning at the next frame boundary. Boundary: a call of vpvk_frame_boundary once
// the host has called it at least once (META / END frame_unit=host: a "frame" of a capture, and of VPVK_CAPTURE_FRAME /
// VPVK_CAPTURE_FRAMES, is what lies between two calls); otherwise a vkQueueSubmit that carries command buffers (the frame the
// vk.perf / vk.hitch lines count; frame_unit=submit: one such submit, not a game frame). The capture is of the device that makes that submit; the other
// devices' objects and calls stay out (END other_device_*). END invalid=1 reason=... says the file is not exact (images over
// VPVK_CAPTURE_IMG_BUDGET_MB, MB per submit, default 512, 0 = no limit, wait for the next submit; calls the file cannot hold). The capture starts when that submit returns (the GPU is drained
// and the state written), so the frames captured are the ones submitted after it; a command buffer already begun then is
// dropped from the capture (counted as cbs_dropped in "vk.capture done"). `path`: a file; a directory (existing, or ending in '/') gets
// vpcap_<unix time>_<pid>.vpcap; null = $VPVK_CAPTURE_DIR (default ".") with that name. Returns 1 queued, 0 a capture is already
// pending or running, -1 tracking is off / no device is tracked.
// Environment: VPVK_CAPTURE_FRAME=<n> starts a capture at the n-th frame (n >= 1; 0 = off) of VPVK_CAPTURE_FRAMES frames
// (default 1) into VPVK_CAPTURE_DIR.
int vpvk_capture_request(uint32_t frames, const char* path);
// 1: start tracking object creation (clears what was tracked); 0: stop and drop it. Returns the previous state.
int vpvk_capture_track(int on);
// 0 idle, 1 request pending, 2 capturing.
int vpvk_capture_state(void);
// The host calls this once per game frame (after submitting it). From the first call on the capture takes these calls as the
// frame boundary instead of the submits; the device captured is the one that submitted most since the previous call. A call
// that finds tracking off costs one atomic load.
void vpvk_frame_boundary(void);
#ifdef __cplusplus
}
#endif
