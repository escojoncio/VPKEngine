/* VPKEngine's own OpenGL ES 2/3 + EGL implementation on Vulkan (gpu/gles). Plain C so Klepton
 * (C) and Swift can use it.
 *
 * Host contract
 *   - vpgles_init() once with the Vulkan driver's vkGetInstanceProcAddr (VPVulkan's).
 *   - The guest resolves every GL and EGL entry point through vpgles_proc(name): the EGL ones by
 *     their egl* names, the GL ones by their gl* names (eglGetProcAddress included). Unknown names
 *     return NULL; the host keeps its own fallback for those.
 *   - Window surfaces present through a host callback (vpgles_set_present): the colour image is a
 *     VkImage of the shared VkDevice, stored bottom-up (GL row 0 = memory row 0): show it flipped.
 *   - OVRPlugin / OpenXR GLES swapchains: the host creates its swapchain images as VkImages of the
 *     shared device (vpgles_device) and hands them to GL as texture names with
 *     vpgles_wrap_image(); the game renders into them with FBOs (2D array + GL_OVR_multiview2 for
 *     the two eyes). The host submits them after vpgles_flush().
 *   - Calling convention: every function takes its arguments the way AAPCS64 lays them out; the
 *     functions that have stack arguments take 64-bit parameters (see gl_api.cpp) so a Darwin
 *     arm64 build reads them where an AAPCS64 caller put them.
 */
#ifndef VPGLES_H
#define VPGLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 on success. Safe to call more than once (later calls return the first result). */
int vpgles_init(void* vk_get_instance_proc_addr);
void vpgles_shutdown(void);

/* GL/EGL entry point by name, or NULL. */
void* vpgles_proc(const char* name);

/* The Vulkan objects the layer renders with (VkInstance, VkPhysicalDevice, VkDevice, VkQueue). */
void vpgles_device(void** instance, void** physical_device, void** device, void** queue);

/* Present callback for window surfaces: `image` is the VkImage (VK_IMAGE_LAYOUT_GENERAL, the
 * colour format of the config), `width`/`height` its size, `window` the ANativeWindow given to
 * eglCreateWindowSurface. Called from eglSwapBuffers after the frame's work was submitted and
 * completed. */
typedef void (*vpgles_present_fn)(void* ctx, void* window, void* image, uint32_t width, uint32_t height, uint32_t vk_format);
void vpgles_set_present(vpgles_present_fn fn, void* ctx);

/* Window size query for eglCreateWindowSurface (ANativeWindow_getWidth/Height of the host). */
typedef int (*vpgles_window_size_fn)(void* ctx, void* window, uint32_t* width, uint32_t* height);
void vpgles_set_window_size(vpgles_window_size_fn fn, void* ctx);

/* Wraps an existing VkImage (created on vpgles_device's device; 2D or 2D array; colour or depth)
 * as a texture of the current context: returns the GL texture name (0 on failure). `target` is
 * GL_TEXTURE_2D or GL_TEXTURE_2D_ARRAY; `vk_format` the image's format. The image stays owned by
 * the caller. Used for OVRPlugin/OpenXR swapchain images. */
uint32_t vpgles_wrap_image(void* vk_image, uint32_t vk_format, uint32_t target, uint32_t width, uint32_t height,
                           uint32_t layers, uint32_t levels);

/* Submits the current context's recorded work; wait = 1 also waits for it. */
void vpgles_flush(int wait);

#ifdef __cplusplus
}
#endif

#endif
