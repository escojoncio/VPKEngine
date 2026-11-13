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

#ifndef VPGLES_API
#define VPGLES_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 on success. Safe to call more than once (later calls return the first result). */
VPGLES_API int vpgles_init(void* vk_get_instance_proc_addr);
VPGLES_API void vpgles_shutdown(void);

/* GL/EGL entry point by name, or NULL. */
VPGLES_API void* vpgles_proc(const char* name);

/* The Vulkan objects the layer renders with (VkInstance, VkPhysicalDevice, VkDevice, VkQueue). NULL before vpgles_init and after
 * vpgles_shutdown, except while a present callback (below) that started before the shutdown is still running: the device it uses is then
 * still standing and is returned (until vpgles_init gives a new one). */
VPGLES_API void vpgles_device(void** instance, void** physical_device, void** device, void** queue);

/* Present callback for window surfaces: `image` is the VkImage (VK_IMAGE_LAYOUT_GENERAL, the
 * colour format of the config), `width`/`height` its size, `window` the ANativeWindow given to
 * eglCreateWindowSurface. Called from eglSwapBuffers after the frame's work was submitted and
 * completed.
 *
 * Present ordering and threads: the callback runs on the thread that called eglSwapBuffers, WITHOUT the layer's lock (so it
 * may wait for other threads, call vpgles_* or GL, or submit to the shared queue; VPVulkan serializes queue submissions with
 * its own submitMutex). Presents from different threads can therefore overlap and their callbacks are not ordered against
 * each other or against GL calls other threads make meanwhile; the image passed is kept alive until the callback returns
 * (eglDestroySurface from another thread does not free it under the callback). vpgles_shutdown does not wait for a callback
 * that is running: the shared VkDevice stays valid until the last such callback returns and is destroyed then, so the host
 * must not use it from the callback after it has returned.
 *
 * What the host must do before the callback returns:
 *   - Its work on `image` (a blit or copy it submitted, a read by a pass it recorded) must be finished or synchronised with
 *     whatever else may destroy the image: the layer releases its reference when the callback returns, and if the surface was
 *     destroyed meanwhile (eglDestroySurface on another thread, eglTerminate) that destroys the VkImage at once, with no wait
 *     for the GPU. Wait for the work yourself (a fence of your submission, vkQueueWaitIdle) before returning, or keep the work
 *     off the image (copy it first).
 *   - The one exception is a callback that returns after vpgles_shutdown ran: the last such callback to return then waits for
 *     the GPU (vkDeviceWaitIdle) before it destroys the images of all of them and the device, so work queued on the shared
 *     queue inside the callback is covered. That wait runs WITHOUT the layer's lock (other threads can call vpgles_* or GL while
 *     it lasts) but on the thread of the last callback, after the host's callback function has returned: the host work it waits
 *     for must not depend on that thread doing anything else first. It runs once, however many callbacks were in flight.
 *     vkDeviceWaitIdle requires external synchronisation of ALL the device's queues (Vulkan spec): while that wait (in
 *     vpgles_shutdown, and in the last callback's return after it) is running, the host must not submit to the shared queue
 *     (or any other queue of the device) from another thread unless its Vulkan driver serialises that against the wait
 *     itself (VPVulkan does, with its submitMutex). */
typedef void (*vpgles_present_fn)(void* ctx, void* window, void* image, uint32_t width, uint32_t height, uint32_t vk_format);
VPGLES_API void vpgles_set_present(vpgles_present_fn fn, void* ctx);

/* Window size query for eglCreateWindowSurface (ANativeWindow_getWidth/Height of the host). */
typedef int (*vpgles_window_size_fn)(void* ctx, void* window, uint32_t* width, uint32_t* height);
VPGLES_API void vpgles_set_window_size(vpgles_window_size_fn fn, void* ctx);

/* Wraps an existing VkImage (created on vpgles_device's device; 2D or 2D array; colour or depth)
 * as a texture of the current context: returns the GL texture name (0 on failure). `target` is
 * GL_TEXTURE_2D or GL_TEXTURE_2D_ARRAY; `vk_format` the image's format. The image stays owned by
 * the caller. Used for OVRPlugin/OpenXR swapchain images. */
VPGLES_API uint32_t vpgles_wrap_image(void* vk_image, uint32_t vk_format, uint32_t target, uint32_t width, uint32_t height,
                           uint32_t layers, uint32_t levels);

/* As vpgles_wrap_image, but the image becomes the storage of the existing texture name `name` (created by
 * the app with glGenTextures/glBindTexture); returns `name`, 0 on failure. */
VPGLES_API uint32_t vpgles_wrap_image_as(uint32_t name, void* vk_image, uint32_t vk_format, uint32_t target, uint32_t width,
                           uint32_t height, uint32_t layers, uint32_t levels);

/* Submits the current context's recorded work; wait = 1 also waits for it. */
VPGLES_API void vpgles_flush(int wait);

#ifdef __cplusplus
}
#endif

#endif
