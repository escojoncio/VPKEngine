// The Vulkan entry points the GLES layer calls, resolved once through the host's
// vkGetInstanceProcAddr (VPVulkan on the visor; VPVulkan on the null platform in tests). The layer
// never links a Vulkan loader.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

namespace vpgles {

#define VPGLES_VK_INSTANCE(X)                \
    X(vkCreateInstance)                      \
    X(vkDestroyInstance)                     \
    X(vkEnumeratePhysicalDevices)            \
    X(vkGetPhysicalDeviceProperties)         \
    X(vkGetPhysicalDeviceMemoryProperties)   \
    X(vkGetPhysicalDeviceFormatProperties)   \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkCreateDevice)                        \
    X(vkGetDeviceProcAddr)

#define VPGLES_VK_DEVICE(X)              \
    X(vkDestroyDevice)                   \
    X(vkGetDeviceQueue)                  \
    X(vkDeviceWaitIdle)                  \
    X(vkQueueSubmit)                     \
    X(vkQueueWaitIdle)                   \
    X(vkAllocateMemory)                  \
    X(vkFreeMemory)                      \
    X(vkMapMemory)                       \
    X(vkUnmapMemory)                     \
    X(vkCreateBuffer)                    \
    X(vkDestroyBuffer)                   \
    X(vkGetBufferMemoryRequirements)     \
    X(vkBindBufferMemory)                \
    X(vkCreateImage)                     \
    X(vkDestroyImage)                    \
    X(vkGetImageMemoryRequirements)      \
    X(vkBindImageMemory)                 \
    X(vkCreateBufferView)                \
    X(vkDestroyBufferView)               \
    X(vkCreateImageView)                 \
    X(vkDestroyImageView)                \
    X(vkCreateSampler)                   \
    X(vkDestroySampler)                  \
    X(vkCreateShaderModule)              \
    X(vkDestroyShaderModule)             \
    X(vkCreateDescriptorSetLayout)       \
    X(vkDestroyDescriptorSetLayout)      \
    X(vkCreatePipelineLayout)            \
    X(vkDestroyPipelineLayout)           \
    X(vkCreateGraphicsPipelines)         \
    X(vkCreateComputePipelines)          \
    X(vkDestroyPipeline)                 \
    X(vkCreateDescriptorPool)            \
    X(vkDestroyDescriptorPool)           \
    X(vkResetDescriptorPool)             \
    X(vkAllocateDescriptorSets)          \
    X(vkUpdateDescriptorSets)            \
    X(vkCreateCommandPool)               \
    X(vkDestroyCommandPool)              \
    X(vkAllocateCommandBuffers)          \
    X(vkFreeCommandBuffers)              \
    X(vkBeginCommandBuffer)              \
    X(vkEndCommandBuffer)                \
    X(vkResetCommandBuffer)              \
    X(vkCreateFence)                     \
    X(vkDestroyFence)                    \
    X(vkWaitForFences)                   \
    X(vkResetFences)                     \
    X(vkGetFenceStatus)                  \
    X(vkCmdBeginRendering)               \
    X(vkCmdEndRendering)                 \
    X(vkCmdBindPipeline)                 \
    X(vkCmdBindDescriptorSets)           \
    X(vkCmdBindVertexBuffers)            \
    X(vkCmdBindIndexBuffer)              \
    X(vkCmdSetViewport)                  \
    X(vkCmdSetScissor)                   \
    X(vkCmdSetStencilReference)          \
    X(vkCmdSetStencilCompareMask)        \
    X(vkCmdSetStencilWriteMask)          \
    X(vkCmdSetBlendConstants)            \
    X(vkCmdSetDepthBias)                 \
    X(vkCmdSetLineWidth)                 \
    X(vkCmdDraw)                         \
    X(vkCmdDrawIndexed)                  \
    X(vkCmdDrawIndirect)                 \
    X(vkCmdDrawIndexedIndirect)          \
    X(vkCmdDispatch)                     \
    X(vkCmdDispatchIndirect)             \
    X(vkCmdPipelineBarrier)              \
    X(vkCreateQueryPool)                 \
    X(vkDestroyQueryPool)                \
    X(vkGetQueryPoolResults)             \
    X(vkCmdResetQueryPool)               \
    X(vkCmdBeginQuery)                   \
    X(vkCmdEndQuery)                     \
    X(vkCmdCopyBuffer)                   \
    X(vkCmdCopyBufferToImage)            \
    X(vkCmdCopyImageToBuffer)            \
    X(vkCmdCopyImage)                    \
    X(vkCmdBlitImage)                    \
    X(vkCmdResolveImage)                 \
    X(vkCmdClearAttachments)             \
    X(vkCmdClearColorImage)              \
    X(vkCmdClearDepthStencilImage)       \
    X(vkCmdFillBuffer)

struct Vk {
#define VPGLES_DECL(f) PFN_##f f = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    VPGLES_VK_INSTANCE(VPGLES_DECL)
    VPGLES_VK_DEVICE(VPGLES_DECL)
#undef VPGLES_DECL
    // VPVulkan's private hook (pipelines.cpp): attachments of the open dynamic-rendering pass undefined when it ends.
    // Optional: null with another driver (invalidated attachments are then stored).
    void (*vpvk_cmd_pass_drop)(VkCommandBuffer, uint32_t colorMask, int depth, int stencil) = nullptr;
    // VPVulkan's private hook (pipelines.cpp): pipeline-recipe counters {loaded, saved, prewarmed, miss_in_game}. Optional.
    void (*vpvk_recipe_stats)(VkDevice, uint32_t out[4]) = nullptr;
    // Loads the global and instance-level functions (instance may be null for the global ones).
    bool loadInstance(PFN_vkGetInstanceProcAddr gipa, VkInstance instance);
    bool loadDevice(VkDevice device);
};

}  // namespace vpgles
