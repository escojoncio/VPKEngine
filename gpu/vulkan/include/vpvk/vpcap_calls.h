// VPCAP call payloads: one symmetrical codec (Cod) writes a Vulkan call's arguments and reads them back, so the capture
// (writer) and vpreplay (reader) share a single definition per call.
//
// Payload of a call = its arguments in prototype order (the VkResult is not stored):
//   arithmetic / enum        raw bytes
//   handle (any Vk handle)   u64 stable id, 0 = null (never an address)
//   output handle (T*)       u64 id the call produced; outputs (T*, n) -> n ids
//   const Struct*            u8 present, then the struct codec S<Struct>
//   const T* + count         u8 present, then count elements (POD raw, handles as ids, structs by S<T>)
//   const char* / char**     u8 present, u32 length, bytes (no NUL)
//   void* user pointers      dropped (VkAllocationCallbacks, mapped pointers, pipeline cache -> null)
// A struct codec stores the struct as its raw LP64 bytes with handle fields replaced by ids and pointer fields
// zeroed, then (in field order) the out-of-line data its pointers referenced, then its pNext chain: u32 sType + the chain
// element (which ends with its own pNext), 0 terminating. pNext structs outside VPCAP_PNEXT are dropped and counted
// (Cod::droppedNext). Decoded pointers live in the Cod's arena, valid while the Cod lives; decoded handle values are ids
// cast to the handle type (reinterpret_cast<uint64_t>).
#pragma once

#include <vulkan/vulkan_core.h>

#include <cstring>
#include <deque>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "vpvk/vpcap.h"

namespace vpcap {

// Writer side: the stable id of a live handle (assigned on first sight).
struct HandleIds {
  virtual uint64_t idOf(const void* handle) = 0;

 protected:
  ~HandleIds() = default;
};

template <class T, class = void>
struct IsComplete : std::false_type {};
template <class T>
struct IsComplete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};
// VK_DEFINE_(NON_)DISPATCHABLE_HANDLE: a pointer to an incomplete struct (void* is not a handle).
template <class T>
constexpr bool kIsHandle = std::is_pointer_v<T> && !std::is_void_v<std::remove_cv_t<std::remove_pointer_t<T>>> &&
                           !IsComplete<std::remove_cv_t<std::remove_pointer_t<T>>>::value;

template <class T>
struct S;  // struct codecs, specialised below

#define VPCAP_PNEXT(X)                                                                                                 \
  X(VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO, VkRenderPassMultiviewCreateInfo)                              \
  X(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, VkPipelineRenderingCreateInfo)                                   \
  X(VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, VkSemaphoreTypeCreateInfo)                                           \
  X(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, VkMemoryAllocateFlagsInfo)                                           \
  X(VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, VkImageViewUsageCreateInfo)                                        \
  X(VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, VkImageFormatListCreateInfo)                                      \
  X(VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, VkTimelineSemaphoreSubmitInfo)                                   \
  X(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK, VkWriteDescriptorSetInlineUniformBlock)               \
  X(VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE, VkSubpassDescriptionDepthStencilResolve)              \
  X(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, VkDescriptorSetLayoutBindingFlagsCreateInfo)    \
  X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, VkPhysicalDeviceFeatures2)                                           \
  X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, VkPhysicalDeviceVulkan11Features)                           \
  X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features)                           \
  X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features)

struct Cod {
  explicit Cod(HandleIds* ids) : w(true), ids(ids) {}                                    // writer
  Cod(const uint8_t* p, size_t n) : w(false), cur(p), end(p + n) {}                     // reader
  bool w;
  HandleIds* ids = nullptr;
  std::vector<uint8_t> out;                // writer: the payload
  const uint8_t* cur = nullptr;            // reader cursor
  const uint8_t* end = nullptr;
  bool bad = false;                        // reader: payload shorter than the codec expects
  uint32_t droppedNext = 0;                // writer: pNext structs not covered
  std::deque<std::vector<uint64_t>> arena;  // reader: storage of decoded arrays / structs (8-byte aligned)

  // ---- bytes ----
  size_t remaining() const { return size_t(end - cur); }
  void rawBytes(void* x, size_t n) {
    if (w) {
      const auto* b = static_cast<const uint8_t*>(x);
      out.insert(out.end(), b, b + n);
    } else if (remaining() < n) {
      bad = true;
      std::memset(x, 0, n);
    } else {
      std::memcpy(x, cur, n);
      cur += n;
    }
  }
  template <class T>
  void raw(T& x) {
    static_assert(std::is_trivially_copyable_v<T>);
    rawBytes(&x, sizeof(T));
  }
  template <class T>
  T* alloc(size_t n) {
    arena.emplace_back((n * sizeof(T) + 7) / 8 + 1, 0);
    return reinterpret_cast<T*>(arena.back().data());
  }
  bool present(const void* p) {
    uint8_t b = p ? 1 : 0;
    raw(b);
    return b != 0;
  }

  // ---- writer: zero a pointer field of the struct copy just stored (found by address) ----
  struct Frame {
    const char* base;
    size_t size, pos;
  };
  std::vector<Frame> frames;
  void zero(const void* field) {
    if (!w) return;
    const char* a = static_cast<const char*>(field);
    for (size_t i = frames.size(); i--;)
      if (a >= frames[i].base && a < frames[i].base + frames[i].size) {
        std::memset(out.data() + frames[i].pos + size_t(a - frames[i].base), 0, sizeof(void*));
        return;
      }
  }
  // The struct's own bytes. Handle fields must have gone through h() before; pointer fields are zeroed by the
  // pod / ss / sp / hs / str / strs / bytes / next call that stores their data.
  template <class T>
  void st(T& x) {
    if (w) frames.push_back({reinterpret_cast<const char*>(&x), sizeof(T), out.size()});
    raw(x);
  }

  // ---- values ----
  template <class T>
  std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>> v(T& x) {
    raw(x);
  }
  template <class H>
  std::enable_if_t<kIsHandle<H>> v(H& x) {  // a handle argument
    uint64_t i = w ? ids->idOf(x) : 0;
    raw(i);
    if (!w) x = reinterpret_cast<H>(static_cast<uintptr_t>(i));
  }
  template <class H>
  std::enable_if_t<kIsHandle<H>> h(H& x) {  // a handle field of a struct (before st())
    if (w) x = reinterpret_cast<H>(static_cast<uintptr_t>(ids->idOf(x)));
  }
  template <class T>
  void drop(T& x) {
    if (!w) x = T{};
  }
  template <class H>
  void out1(H*& p) {  // output handle
    uint64_t i = (w && p) ? ids->idOf(*p) : 0;
    raw(i);
    if (!w) {
      H* a = alloc<H>(1);
      *a = reinterpret_cast<H>(static_cast<uintptr_t>(i));
      p = a;
    }
  }
  template <class H>
  void outs(H*& p, uint32_t n) {
    const bool pr = present(p && n ? p : nullptr);
    if (!pr) {
      if (!w) p = nullptr;
      return;
    }
    if (!w && remaining() / 8 < n) return void(bad = true), void(p = nullptr);
    H* a = w ? p : alloc<H>(n);
    for (uint32_t k = 0; k < n; ++k) {
      uint64_t i = w ? ids->idOf(p[k]) : 0;
      raw(i);
      if (!w) a[k] = reinterpret_cast<H>(static_cast<uintptr_t>(i));
    }
    if (!w) p = a;
  }

  // ---- arrays / pointers ----
  template <class T>
  void pod(const T*& p, uint32_t n) {
    const bool pr = present(p && n ? p : nullptr);
    if (w) {
      if (pr) rawBytes(const_cast<T*>(p), size_t(n) * sizeof(T));
      zero(&p);
      return;
    }
    p = nullptr;
    if (!pr) return;
    if (remaining() / sizeof(T) < n) return void(bad = true);
    T* a = alloc<T>(n);
    rawBytes(a, size_t(n) * sizeof(T));
    p = a;
  }
  template <class T>
  void bytes(const T*& p, size_t n) {  // pData / pCode: n is a byte count
    const bool pr = present(p && n ? p : nullptr);
    if (w) {
      if (pr) rawBytes(const_cast<T*>(p), n);
      zero(&p);
      return;
    }
    p = nullptr;
    if (!pr) return;
    if (remaining() < n) return void(bad = true);
    auto* a = alloc<uint8_t>(n);
    rawBytes(a, n);
    p = reinterpret_cast<const T*>(a);
  }
  template <class H>
  void hs(const H*& p, uint32_t n) {  // array of handles
    const bool pr = present(p && n ? p : nullptr);
    if (w) {
      if (pr)
        for (uint32_t k = 0; k < n; ++k) {
          uint64_t i = ids->idOf(p[k]);
          raw(i);
        }
      zero(&p);
      return;
    }
    p = nullptr;
    if (!pr) return;
    if (remaining() / 8 < n) return void(bad = true);
    H* a = alloc<H>(n);
    for (uint32_t k = 0; k < n; ++k) {
      uint64_t i = 0;
      raw(i);
      a[k] = reinterpret_cast<H>(static_cast<uintptr_t>(i));
    }
    p = a;
  }
  template <class T>
  void ss(const T*& p, uint32_t n) {  // array of structs with a codec
    const bool pr = present(p && n ? p : nullptr);
    if (w) {
      if (pr)
        for (uint32_t k = 0; k < n; ++k) {
          T el = p[k];
          const size_t mark = frames.size();
          S<T>::run(*this, el);
          frames.resize(mark);
        }
      zero(&p);
      return;
    }
    p = nullptr;
    if (!pr) return;
    if (remaining() < n) return void(bad = true);
    T* a = alloc<T>(n);
    for (uint32_t k = 0; k < n && !bad; ++k) S<T>::run(*this, a[k]);
    p = a;
  }
  template <class T>
  void sp(const T*& p) {  // one optional struct
    const bool pr = present(p);
    if (w) {
      if (pr) {
        T el = *p;
        const size_t mark = frames.size();
        S<T>::run(*this, el);
        frames.resize(mark);
      }
      zero(&p);
      return;
    }
    p = nullptr;
    if (!pr) return;
    T* a = alloc<T>(1);
    S<T>::run(*this, *a);
    p = a;
  }
  void str(const char*& s) {
    const bool pr = present(s);
    if (w) {
      if (pr) {
        uint32_t n = static_cast<uint32_t>(std::strlen(s));
        raw(n);
        rawBytes(const_cast<char*>(s), n);
      }
      zero(&s);
      return;
    }
    s = nullptr;
    if (!pr) return;
    uint32_t n = 0;
    raw(n);
    if (remaining() < n) return void(bad = true);
    char* a = alloc<char>(size_t(n) + 1);
    rawBytes(a, n);
    s = a;
  }
  void strs(const char* const*& pp, uint32_t n) {
    const bool pr = present(pp && n ? pp : nullptr);
    if (w) {
      if (pr)
        for (uint32_t k = 0; k < n; ++k) {
          const char* s = pp[k];
          str(s);
        }
      zero(&pp);
      return;
    }
    pp = nullptr;
    if (!pr) return;
    if (remaining() < n) return void(bad = true);
    const char** a = alloc<const char*>(n);
    for (uint32_t k = 0; k < n; ++k) str(a[k]);
    pp = a;
  }
  void next(const void*& pn);  // defined after the struct codecs
  void next(void*& pn) { next(const_cast<const void*&>(pn)); }  // the feature structs' pNext
};

// ---- struct codecs (leaves first: a codec instantiates the ones it uses) ----
#define VPCAP_POD(T)                                       \
  template <>                                              \
  struct S<T> {                                            \
    static void run(Cod& c, T& x) { c.st(x); }             \
  };
VPCAP_POD(VkPhysicalDeviceFeatures)
VPCAP_POD(VkAttachmentReference)
VPCAP_POD(VkClearColorValue)
VPCAP_POD(VkClearDepthStencilValue)

// Structs that are a plain struct + pNext.
#define VPCAP_PLAIN(T)                                     \
  template <>                                              \
  struct S<T> {                                            \
    static void run(Cod& c, T& x) {                        \
      c.st(x);                                             \
      c.next(x.pNext);                                     \
    }                                                      \
  };
VPCAP_PLAIN(VkMemoryAllocateInfo)
VPCAP_PLAIN(VkSamplerCreateInfo)
VPCAP_PLAIN(VkCommandPoolCreateInfo)
VPCAP_PLAIN(VkFenceCreateInfo)
VPCAP_PLAIN(VkSemaphoreCreateInfo)
VPCAP_PLAIN(VkEventCreateInfo)
VPCAP_PLAIN(VkAttachmentDescription2)
VPCAP_PLAIN(VkAttachmentReference2)
VPCAP_PLAIN(VkSubpassDependency2)
VPCAP_PLAIN(VkSubpassBeginInfo)
VPCAP_PLAIN(VkSubpassEndInfo)
VPCAP_PLAIN(VkMemoryBarrier)
VPCAP_PLAIN(VkPipelineInputAssemblyStateCreateInfo)
VPCAP_PLAIN(VkPipelineTessellationStateCreateInfo)
VPCAP_PLAIN(VkPipelineRasterizationStateCreateInfo)
VPCAP_PLAIN(VkPipelineDepthStencilStateCreateInfo)
VPCAP_PLAIN(VkSemaphoreTypeCreateInfo)
VPCAP_PLAIN(VkMemoryAllocateFlagsInfo)
VPCAP_PLAIN(VkImageViewUsageCreateInfo)
VPCAP_PLAIN(VkPhysicalDeviceFeatures2)  // the features structs: bools only
VPCAP_PLAIN(VkPhysicalDeviceVulkan11Features)
VPCAP_PLAIN(VkPhysicalDeviceVulkan12Features)
VPCAP_PLAIN(VkPhysicalDeviceVulkan13Features)
#undef VPCAP_PLAIN

#define SC(T)         \
  template <>         \
  struct S<T> {       \
    static void run(Cod& c, T& x)

SC(VkApplicationInfo) {
  c.st(x);
  c.str(x.pApplicationName);
  c.str(x.pEngineName);
  c.next(x.pNext);
}
};
SC(VkInstanceCreateInfo) {
  c.st(x);
  c.sp(x.pApplicationInfo);
  c.strs(x.ppEnabledLayerNames, x.enabledLayerCount);
  c.strs(x.ppEnabledExtensionNames, x.enabledExtensionCount);
  c.next(x.pNext);
}
};
SC(VkDeviceQueueCreateInfo) {
  c.st(x);
  c.pod(x.pQueuePriorities, x.queueCount);
  c.next(x.pNext);
}
};
SC(VkDeviceCreateInfo) {
  c.st(x);
  c.ss(x.pQueueCreateInfos, x.queueCreateInfoCount);
  c.strs(x.ppEnabledLayerNames, x.enabledLayerCount);
  c.strs(x.ppEnabledExtensionNames, x.enabledExtensionCount);
  c.sp(x.pEnabledFeatures);
  c.next(x.pNext);
}
};
SC(VkBufferCreateInfo) {
  c.st(x);
  c.pod(x.pQueueFamilyIndices, x.queueFamilyIndexCount);
  c.next(x.pNext);
}
};
SC(VkImageFormatListCreateInfo) {
  c.st(x);
  c.pod(x.pViewFormats, x.viewFormatCount);
  c.next(x.pNext);
}
};
SC(VkImageCreateInfo) {
  c.st(x);
  c.pod(x.pQueueFamilyIndices, x.queueFamilyIndexCount);
  c.next(x.pNext);
}
};
SC(VkImageViewCreateInfo) {
  c.h(x.image);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkBufferViewCreateInfo) {
  c.h(x.buffer);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkDescriptorSetLayoutBinding) {
  if (c.w && x.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER && x.descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
    x.pImmutableSamplers = nullptr;  // ignored for the other types
  c.st(x);
  c.hs(x.pImmutableSamplers, x.descriptorCount);
}
};
SC(VkDescriptorSetLayoutBindingFlagsCreateInfo) {
  c.st(x);
  c.pod(x.pBindingFlags, x.bindingCount);
  c.next(x.pNext);
}
};
SC(VkDescriptorSetLayoutCreateInfo) {
  c.st(x);
  c.ss(x.pBindings, x.bindingCount);
  c.next(x.pNext);
}
};
SC(VkPipelineLayoutCreateInfo) {
  c.st(x);
  c.hs(x.pSetLayouts, x.setLayoutCount);
  c.pod(x.pPushConstantRanges, x.pushConstantRangeCount);
  c.next(x.pNext);
}
};
SC(VkDescriptorPoolCreateInfo) {
  c.st(x);
  c.pod(x.pPoolSizes, x.poolSizeCount);
  c.next(x.pNext);
}
};
SC(VkSubpassDescription) {
  c.st(x);
  c.pod(x.pInputAttachments, x.inputAttachmentCount);
  c.pod(x.pColorAttachments, x.colorAttachmentCount);
  c.pod(x.pResolveAttachments, x.colorAttachmentCount);
  c.sp(x.pDepthStencilAttachment);
  c.pod(x.pPreserveAttachments, x.preserveAttachmentCount);
}
};
SC(VkRenderPassMultiviewCreateInfo) {
  c.st(x);
  c.pod(x.pViewMasks, x.subpassCount);
  c.pod(x.pViewOffsets, x.dependencyCount);
  c.pod(x.pCorrelationMasks, x.correlationMaskCount);
  c.next(x.pNext);
}
};
SC(VkRenderPassCreateInfo) {
  c.st(x);
  c.pod(x.pAttachments, x.attachmentCount);
  c.ss(x.pSubpasses, x.subpassCount);
  c.pod(x.pDependencies, x.dependencyCount);
  c.next(x.pNext);
}
};
SC(VkSubpassDescriptionDepthStencilResolve) {
  c.st(x);
  c.sp(x.pDepthStencilResolveAttachment);
  c.next(x.pNext);
}
};
SC(VkSubpassDescription2) {
  c.st(x);
  c.ss(x.pInputAttachments, x.inputAttachmentCount);
  c.ss(x.pColorAttachments, x.colorAttachmentCount);
  c.ss(x.pResolveAttachments, x.colorAttachmentCount);
  c.sp(x.pDepthStencilAttachment);
  c.pod(x.pPreserveAttachments, x.preserveAttachmentCount);
  c.next(x.pNext);
}
};
SC(VkRenderPassCreateInfo2) {
  c.st(x);
  c.ss(x.pAttachments, x.attachmentCount);
  c.ss(x.pSubpasses, x.subpassCount);
  c.ss(x.pDependencies, x.dependencyCount);
  c.pod(x.pCorrelatedViewMasks, x.correlatedViewMaskCount);
  c.next(x.pNext);
}
};
SC(VkFramebufferCreateInfo) {
  c.h(x.renderPass);
  c.st(x);
  c.hs(x.pAttachments, x.attachmentCount);
  c.next(x.pNext);
}
};
SC(VkShaderModuleCreateInfo) {
  c.st(x);
  c.bytes(x.pCode, x.codeSize);
  c.next(x.pNext);
}
};
SC(VkSpecializationInfo) {
  c.st(x);
  c.pod(x.pMapEntries, x.mapEntryCount);
  c.bytes(x.pData, x.dataSize);
}
};
SC(VkPipelineShaderStageCreateInfo) {
  c.h(x.module);
  c.st(x);
  c.str(x.pName);
  c.sp(x.pSpecializationInfo);
  c.next(x.pNext);
}
};
SC(VkPipelineVertexInputStateCreateInfo) {
  c.st(x);
  c.pod(x.pVertexBindingDescriptions, x.vertexBindingDescriptionCount);
  c.pod(x.pVertexAttributeDescriptions, x.vertexAttributeDescriptionCount);
  c.next(x.pNext);
}
};
SC(VkPipelineViewportStateCreateInfo) {
  c.st(x);
  c.pod(x.pViewports, x.viewportCount);
  c.pod(x.pScissors, x.scissorCount);
  c.next(x.pNext);
}
};
SC(VkPipelineMultisampleStateCreateInfo) {
  c.st(x);
  c.pod(x.pSampleMask, (static_cast<uint32_t>(x.rasterizationSamples) + 31) / 32);
  c.next(x.pNext);
}
};
SC(VkPipelineColorBlendStateCreateInfo) {
  c.st(x);
  c.pod(x.pAttachments, x.attachmentCount);
  c.next(x.pNext);
}
};
SC(VkPipelineDynamicStateCreateInfo) {
  c.st(x);
  c.pod(x.pDynamicStates, x.dynamicStateCount);
  c.next(x.pNext);
}
};
SC(VkPipelineRenderingCreateInfo) {
  c.st(x);
  c.pod(x.pColorAttachmentFormats, x.colorAttachmentCount);
  c.next(x.pNext);
}
};
SC(VkGraphicsPipelineCreateInfo) {
  c.h(x.layout);
  c.h(x.renderPass);
  c.h(x.basePipelineHandle);
  c.st(x);
  c.ss(x.pStages, x.stageCount);
  c.sp(x.pVertexInputState);
  c.sp(x.pInputAssemblyState);
  c.sp(x.pTessellationState);
  c.sp(x.pViewportState);
  c.sp(x.pRasterizationState);
  c.sp(x.pMultisampleState);
  c.sp(x.pDepthStencilState);
  c.sp(x.pColorBlendState);
  c.sp(x.pDynamicState);
  c.next(x.pNext);
}
};
SC(VkComputePipelineCreateInfo) {
  c.h(x.stage.module);
  c.h(x.layout);
  c.h(x.basePipelineHandle);
  c.st(x);
  c.str(x.stage.pName);
  c.sp(x.stage.pSpecializationInfo);
  c.next(x.stage.pNext);
  c.next(x.pNext);
}
};
SC(VkCommandBufferAllocateInfo) {
  c.h(x.commandPool);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkDescriptorSetAllocateInfo) {
  c.h(x.descriptorPool);
  c.st(x);
  c.hs(x.pSetLayouts, x.descriptorSetCount);
  c.next(x.pNext);
}
};
SC(VkCommandBufferInheritanceInfo) {
  c.h(x.renderPass);
  c.h(x.framebuffer);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkCommandBufferBeginInfo) {
  c.st(x);
  c.sp(x.pInheritanceInfo);
  c.next(x.pNext);
}
};
SC(VkRenderPassBeginInfo) {
  c.h(x.renderPass);
  c.h(x.framebuffer);
  c.st(x);
  c.pod(x.pClearValues, x.clearValueCount);
  c.next(x.pNext);
}
};
SC(VkRenderingAttachmentInfo) {
  c.h(x.imageView);
  c.h(x.resolveImageView);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkRenderingInfo) {
  c.st(x);
  c.ss(x.pColorAttachments, x.colorAttachmentCount);
  c.sp(x.pDepthAttachment);
  c.sp(x.pStencilAttachment);
  c.next(x.pNext);
}
};
SC(VkBufferMemoryBarrier) {
  c.h(x.buffer);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkImageMemoryBarrier) {
  c.h(x.image);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkDescriptorImageInfo) {
  c.h(x.sampler);
  c.h(x.imageView);
  c.st(x);
}
};
SC(VkDescriptorBufferInfo) {
  c.h(x.buffer);
  c.st(x);
}
};
SC(VkWriteDescriptorSetInlineUniformBlock) {
  c.st(x);
  c.bytes(x.pData, x.dataSize);
  c.next(x.pNext);
}
};
SC(VkWriteDescriptorSet) {
  if (c.w) {  // the pointer that matches the type; the others are ignored by Vulkan and may be garbage
    switch (x.descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        x.pBufferInfo = nullptr, x.pTexelBufferView = nullptr;
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        x.pImageInfo = nullptr, x.pTexelBufferView = nullptr;
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        x.pImageInfo = nullptr, x.pBufferInfo = nullptr;
        break;
      default:
        x.pImageInfo = nullptr, x.pBufferInfo = nullptr, x.pTexelBufferView = nullptr;
    }
  }
  c.h(x.dstSet);
  c.st(x);
  c.ss(x.pImageInfo, x.descriptorCount);
  c.ss(x.pBufferInfo, x.descriptorCount);
  c.hs(x.pTexelBufferView, x.descriptorCount);
  c.next(x.pNext);
}
};
SC(VkCopyDescriptorSet) {
  c.h(x.srcSet);
  c.h(x.dstSet);
  c.st(x);
  c.next(x.pNext);
}
};
SC(VkTimelineSemaphoreSubmitInfo) {
  c.st(x);
  c.pod(x.pWaitSemaphoreValues, x.waitSemaphoreValueCount);
  c.pod(x.pSignalSemaphoreValues, x.signalSemaphoreValueCount);
  c.next(x.pNext);
}
};
SC(VkSubmitInfo) {
  c.st(x);
  c.hs(x.pWaitSemaphores, x.waitSemaphoreCount);
  c.pod(x.pWaitDstStageMask, x.waitSemaphoreCount);
  c.hs(x.pCommandBuffers, x.commandBufferCount);
  c.hs(x.pSignalSemaphores, x.signalSemaphoreCount);
  c.next(x.pNext);
}
};
SC(VkMappedMemoryRange) {
  c.h(x.memory);
  c.st(x);
  c.next(x.pNext);
}
};
#undef SC

inline void Cod::next(const void*& pn) {
  if (w) {
    for (const auto* n = static_cast<const VkBaseInStructure*>(pn); n; n = n->pNext) {
      switch (n->sType) {
#define VPCAP_X(stype, T)                                \
  case stype: {                                          \
    uint32_t t = stype;                                  \
    raw(t);                                              \
    T el = *reinterpret_cast<const T*>(n);               \
    const size_t mark = frames.size();                   \
    S<T>::run(*this, el);                                \
    frames.resize(mark);                                 \
    zero(&pn);                                           \
    return;                                              \
  }
        VPCAP_PNEXT(VPCAP_X)
#undef VPCAP_X
        default:
          ++droppedNext;
      }
    }
    uint32_t t = 0;
    raw(t);
    zero(&pn);
    return;
  }
  uint32_t t = 0;
  raw(t);
  pn = nullptr;
  if (!t || bad) return;
  switch (t) {
#define VPCAP_X(stype, T)          \
  case stype: {                    \
    T* a = alloc<T>(1);            \
    S<T>::run(*this, *a);          \
    pn = a;                        \
    return;                        \
  }
    VPCAP_PNEXT(VPCAP_X)
#undef VPCAP_X
    default:
      bad = true;
  }
}

// ---- calls ----
template <uint16_t ID>
struct CallDef;  // { using Fn = PFN_<name>; static const char* kName; template <A> static void cod(Cod&, A&); }
template <class Fn>
struct FnArgs;
template <class R, class... P>
struct FnArgs<R(VKAPI_PTR*)(P...)> {
  using type = std::tuple<P...>;
};
template <uint16_t ID>
using ArgsOf = typename FnArgs<typename CallDef<ID>::Fn>::type;

#define VPCAP_UNPAREN(...) __VA_ARGS__
// A call with its own argument list: `names` binds the tuple, the rest codes each argument.
#define VPCAP_CALL(name, names, ...)                                                   \
  template <>                                                                          \
  struct CallDef<id::name> {                                                           \
    using Fn = PFN_##name;                                                             \
    static constexpr const char* kName = #name;                                        \
    template <class A>                                                                 \
    static void cod(Cod& c, A& a) {                                                    \
      auto& [VPCAP_UNPAREN names] = a;                                                 \
      __VA_ARGS__                                                                      \
    }                                                                                  \
  };
// Only arithmetic / enum / handle arguments.
#define VPCAP_ALL(name)                                                                \
  template <>                                                                          \
  struct CallDef<id::name> {                                                           \
    using Fn = PFN_##name;                                                             \
    static constexpr const char* kName = #name;                                        \
    template <class A>                                                                 \
    static void cod(Cod& c, A& a) {                                                    \
      std::apply([&](auto&... x) { (c.v(x), ...); }, a);                               \
    }                                                                                  \
  };
#define VPCAP_CREATE(name) VPCAP_CALL(name, (dev, ci, ac, out), c.v(dev); c.sp(ci); c.drop(ac); c.out1(out);)
#define VPCAP_DESTROY(name) VPCAP_CALL(name, (dev, obj, ac), c.v(dev); c.v(obj); c.drop(ac);)

VPCAP_CALL(vkCreateInstance, (ci, ac, out), c.sp(ci); c.drop(ac); c.out1(out);)
VPCAP_CALL(vkDestroyInstance, (inst, ac), c.v(inst); c.drop(ac);)
VPCAP_CALL(vkCreateDevice, (pd, ci, ac, out), c.v(pd); c.sp(ci); c.drop(ac); c.out1(out);)
VPCAP_CALL(vkDestroyDevice, (dev, ac), c.v(dev); c.drop(ac);)
VPCAP_CALL(vkGetDeviceQueue, (dev, family, index, out), c.v(dev); c.v(family); c.v(index); c.out1(out);)
VPCAP_CREATE(vkAllocateMemory)
VPCAP_DESTROY(vkFreeMemory)
VPCAP_CALL(vkMapMemory, (dev, mem, offset, size, flags, pp), c.v(dev); c.v(mem); c.v(offset); c.v(size); c.v(flags); c.drop(pp);)
VPCAP_ALL(vkUnmapMemory)
VPCAP_CALL(vkFlushMappedMemoryRanges, (dev, n, ranges), c.v(dev); c.v(n); c.ss(ranges, n);)
VPCAP_CREATE(vkCreateBuffer)
VPCAP_DESTROY(vkDestroyBuffer)
VPCAP_ALL(vkBindBufferMemory)
VPCAP_CREATE(vkCreateBufferView)
VPCAP_DESTROY(vkDestroyBufferView)
VPCAP_CREATE(vkCreateImage)
VPCAP_DESTROY(vkDestroyImage)
VPCAP_ALL(vkBindImageMemory)
VPCAP_CREATE(vkCreateImageView)
VPCAP_DESTROY(vkDestroyImageView)
VPCAP_CREATE(vkCreateSampler)
VPCAP_DESTROY(vkDestroySampler)
VPCAP_CREATE(vkCreateDescriptorSetLayout)
VPCAP_DESTROY(vkDestroyDescriptorSetLayout)
VPCAP_CREATE(vkCreatePipelineLayout)
VPCAP_DESTROY(vkDestroyPipelineLayout)
VPCAP_CREATE(vkCreateDescriptorPool)
VPCAP_DESTROY(vkDestroyDescriptorPool)
VPCAP_ALL(vkResetDescriptorPool)
VPCAP_CALL(vkAllocateDescriptorSets, (dev, ai, sets), c.v(dev); c.sp(ai); c.outs(sets, ai ? ai->descriptorSetCount : 0);)
VPCAP_CALL(vkFreeDescriptorSets, (dev, pool, n, sets), c.v(dev); c.v(pool); c.v(n); c.hs(sets, n);)
VPCAP_CALL(vkUpdateDescriptorSets, (dev, wn, writes, cn, copies),
           c.v(dev); c.v(wn); c.ss(writes, wn); c.v(cn); c.ss(copies, cn);)
VPCAP_CREATE(vkCreateRenderPass)
VPCAP_CREATE(vkCreateRenderPass2)
VPCAP_DESTROY(vkDestroyRenderPass)
VPCAP_CREATE(vkCreateFramebuffer)
VPCAP_DESTROY(vkDestroyFramebuffer)
VPCAP_CREATE(vkCreateShaderModule)
VPCAP_DESTROY(vkDestroyShaderModule)
VPCAP_CALL(vkCreateGraphicsPipelines, (dev, cache, n, infos, ac, pipes),
           c.v(dev); c.drop(cache); c.v(n); c.ss(infos, n); c.drop(ac); c.outs(pipes, n);)
VPCAP_CALL(vkCreateComputePipelines, (dev, cache, n, infos, ac, pipes),
           c.v(dev); c.drop(cache); c.v(n); c.ss(infos, n); c.drop(ac); c.outs(pipes, n);)
VPCAP_DESTROY(vkDestroyPipeline)
VPCAP_CREATE(vkCreateCommandPool)
VPCAP_DESTROY(vkDestroyCommandPool)
VPCAP_ALL(vkResetCommandPool)
VPCAP_CALL(vkAllocateCommandBuffers, (dev, ai, cbs), c.v(dev); c.sp(ai); c.outs(cbs, ai ? ai->commandBufferCount : 0);)
VPCAP_CALL(vkFreeCommandBuffers, (dev, pool, n, cbs), c.v(dev); c.v(pool); c.v(n); c.hs(cbs, n);)
VPCAP_CALL(vkBeginCommandBuffer, (cb, bi), c.v(cb); c.sp(bi);)
VPCAP_ALL(vkEndCommandBuffer)
VPCAP_ALL(vkResetCommandBuffer)
VPCAP_CREATE(vkCreateFence)
VPCAP_DESTROY(vkDestroyFence)
VPCAP_CALL(vkResetFences, (dev, n, fences), c.v(dev); c.v(n); c.hs(fences, n);)
VPCAP_CALL(vkWaitForFences, (dev, n, fences, all, timeout), c.v(dev); c.v(n); c.hs(fences, n); c.v(all); c.v(timeout);)
VPCAP_CREATE(vkCreateSemaphore)
VPCAP_DESTROY(vkDestroySemaphore)
VPCAP_CREATE(vkCreateEvent)
VPCAP_DESTROY(vkDestroyEvent)
VPCAP_CALL(vkQueueSubmit, (q, n, infos, fence), c.v(q); c.v(n); c.ss(infos, n); c.v(fence);)
VPCAP_ALL(vkQueueWaitIdle)
VPCAP_ALL(vkDeviceWaitIdle)

VPCAP_CALL(vkCmdBeginRenderPass, (cb, bi, contents), c.v(cb); c.sp(bi); c.v(contents);)
VPCAP_CALL(vkCmdBeginRenderPass2, (cb, bi, sbi), c.v(cb); c.sp(bi); c.sp(sbi);)
VPCAP_ALL(vkCmdEndRenderPass)
VPCAP_CALL(vkCmdEndRenderPass2, (cb, sei), c.v(cb); c.sp(sei);)
VPCAP_ALL(vkCmdNextSubpass)
VPCAP_CALL(vkCmdBeginRendering, (cb, ri), c.v(cb); c.sp(ri);)
VPCAP_ALL(vkCmdEndRendering)
VPCAP_ALL(vkCmdBindPipeline)
VPCAP_CALL(vkCmdBindDescriptorSets, (cb, bp, layout, first, n, sets, dn, dyn),
           c.v(cb); c.v(bp); c.v(layout); c.v(first); c.v(n); c.hs(sets, n); c.v(dn); c.pod(dyn, dn);)
VPCAP_CALL(vkCmdPushConstants, (cb, layout, stages, offset, size, values),
           c.v(cb); c.v(layout); c.v(stages); c.v(offset); c.v(size); c.bytes(values, size);)
VPCAP_CALL(vkCmdBindVertexBuffers, (cb, first, n, buffers, offsets),
           c.v(cb); c.v(first); c.v(n); c.hs(buffers, n); c.pod(offsets, n);)
VPCAP_CALL(vkCmdBindVertexBuffers2, (cb, first, n, buffers, offsets, sizes, strides),
           c.v(cb); c.v(first); c.v(n); c.hs(buffers, n); c.pod(offsets, n); c.pod(sizes, n); c.pod(strides, n);)
VPCAP_ALL(vkCmdBindIndexBuffer)
VPCAP_CALL(vkCmdSetViewport, (cb, first, n, p), c.v(cb); c.v(first); c.v(n); c.pod(p, n);)
VPCAP_CALL(vkCmdSetScissor, (cb, first, n, p), c.v(cb); c.v(first); c.v(n); c.pod(p, n);)
VPCAP_CALL(vkCmdSetViewportWithCount, (cb, n, p), c.v(cb); c.v(n); c.pod(p, n);)
VPCAP_CALL(vkCmdSetScissorWithCount, (cb, n, p), c.v(cb); c.v(n); c.pod(p, n);)
VPCAP_ALL(vkCmdSetStencilReference)
VPCAP_CALL(vkCmdSetBlendConstants, (cb, bc), c.v(cb); c.pod(bc, 4);)
VPCAP_ALL(vkCmdSetLineWidth)
VPCAP_ALL(vkCmdSetDepthBias)
VPCAP_ALL(vkCmdSetDepthBounds)
VPCAP_ALL(vkCmdSetStencilCompareMask)
VPCAP_ALL(vkCmdSetStencilWriteMask)
VPCAP_ALL(vkCmdSetCullMode)
VPCAP_ALL(vkCmdSetFrontFace)
VPCAP_ALL(vkCmdSetPrimitiveTopology)
VPCAP_ALL(vkCmdSetDepthTestEnable)
VPCAP_ALL(vkCmdSetDepthWriteEnable)
VPCAP_ALL(vkCmdSetDepthCompareOp)
VPCAP_ALL(vkCmdSetDepthBoundsTestEnable)
VPCAP_ALL(vkCmdSetStencilTestEnable)
VPCAP_ALL(vkCmdSetStencilOp)
VPCAP_ALL(vkCmdSetRasterizerDiscardEnable)
VPCAP_ALL(vkCmdSetDepthBiasEnable)
VPCAP_ALL(vkCmdSetPrimitiveRestartEnable)
VPCAP_ALL(vkCmdDraw)
VPCAP_ALL(vkCmdDrawIndexed)
VPCAP_ALL(vkCmdDrawIndirect)
VPCAP_ALL(vkCmdDrawIndexedIndirect)
VPCAP_ALL(vkCmdDispatch)
VPCAP_ALL(vkCmdDispatchIndirect)
VPCAP_CALL(vkCmdCopyBuffer, (cb, src, dst, n, r), c.v(cb); c.v(src); c.v(dst); c.v(n); c.pod(r, n);)
VPCAP_CALL(vkCmdCopyImage, (cb, src, sl, dst, dl, n, r), c.v(cb); c.v(src); c.v(sl); c.v(dst); c.v(dl); c.v(n); c.pod(r, n);)
VPCAP_CALL(vkCmdCopyBufferToImage, (cb, src, dst, dl, n, r), c.v(cb); c.v(src); c.v(dst); c.v(dl); c.v(n); c.pod(r, n);)
VPCAP_CALL(vkCmdCopyImageToBuffer, (cb, src, sl, dst, n, r), c.v(cb); c.v(src); c.v(sl); c.v(dst); c.v(n); c.pod(r, n);)
VPCAP_CALL(vkCmdBlitImage, (cb, src, sl, dst, dl, n, r, filter),
           c.v(cb); c.v(src); c.v(sl); c.v(dst); c.v(dl); c.v(n); c.pod(r, n); c.v(filter);)
VPCAP_CALL(vkCmdResolveImage, (cb, src, sl, dst, dl, n, r), c.v(cb); c.v(src); c.v(sl); c.v(dst); c.v(dl); c.v(n); c.pod(r, n);)
VPCAP_CALL(vkCmdClearColorImage, (cb, image, layout, color, n, ranges),
           c.v(cb); c.v(image); c.v(layout); c.sp(color); c.v(n); c.pod(ranges, n);)
VPCAP_CALL(vkCmdClearDepthStencilImage, (cb, image, layout, ds, n, ranges),
           c.v(cb); c.v(image); c.v(layout); c.sp(ds); c.v(n); c.pod(ranges, n);)
VPCAP_CALL(vkCmdClearAttachments, (cb, an, atts, rn, rects), c.v(cb); c.v(an); c.pod(atts, an); c.v(rn); c.pod(rects, rn);)
VPCAP_ALL(vkCmdFillBuffer)
VPCAP_CALL(vkCmdUpdateBuffer, (cb, dst, offset, size, data), c.v(cb); c.v(dst); c.v(offset); c.v(size); c.bytes(data, size);)
VPCAP_CALL(vkCmdPipelineBarrier, (cb, src, dst, dep, mn, mem, bn, buf, in, img),
           c.v(cb); c.v(src); c.v(dst); c.v(dep); c.v(mn); c.ss(mem, mn); c.v(bn); c.ss(buf, bn); c.v(in); c.ss(img, in);)
VPCAP_CALL(vkCmdExecuteCommands, (cb, n, cbs), c.v(cb); c.v(n); c.hs(cbs, n);)

// ---- reading ----
// The arguments of one stored call. `args` holds ids where handles go (reinterpret_cast<uint64_t>); arrays and structs
// point into `cod`'s arena.
template <uint16_t ID>
struct Decoded {
  Decoded(const uint8_t* p, size_t n) : cod(p, n) {}
  Cod cod;
  ArgsOf<ID> args{};
  bool ok = false;  // false: the payload was shorter than the codec expects
};
template <uint16_t ID>
std::unique_ptr<Decoded<ID>> decode(const Call& call) {
  if (call.id != ID) return nullptr;
  auto d = std::make_unique<Decoded<ID>>(call.data, call.size);
  CallDef<ID>::cod(d->cod, d->args);
  d->ok = !d->cod.bad;
  return d;
}
inline uint64_t idOfHandle(const void* decodedHandle) { return reinterpret_cast<uintptr_t>(decodedHandle); }

}  // namespace vpcap
