// Internal EGL objects.
#pragma once

#include <memory>

#include "context.h"

namespace vpgles {

struct Config {
    int id;
    int r, g, b, a, depth, stencil, samples;
    VkFormat color, ds;
};

struct Surface {
    enum Kind { Window, Pbuffer, External } kind = Pbuffer;
    const Config* config = nullptr;
    uint32_t width = 0, height = 0;
    ImageP color;          // colour buffer (memory row 0 = GL row 0)
    ImageP depthStencil;   // optional, single-sample (a multisample config renders it in a transient twin, like `color`)
    uint32_t msSamples = 1;  // multisampled configs: colour and depth/stencil are rendered in transient multisample twins (never stored)
    void* window = nullptr;  // ANativeWindow (opaque)
    uint64_t presented = 0;
    int swapInterval = 1;
    int bound = 0;      // veces que es draw/read actual en algún hilo (Device::lock)
    bool dead = false;  // eglDestroySurface pedido mientras era actual: se borra al soltarse
};

// Creates the images of a surface.
bool createSurfaceImages(Device* dev, Surface& s);

}  // namespace vpgles
