// Metal capabilities of the machine (CI runner), printed once. Build: clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation metal_probe.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <initializer_list>

static void yn(const char* what, BOOL v) { printf("%-46s %s\n", what, v ? "yes" : "no"); }

int main(void) {
  @autoreleasepool {
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    printf("devices=%lu\n", (unsigned long)all.count);
    for (id<MTLDevice> d in all) printf("  device: %s registryID=%llu\n", d.name.UTF8String, d.registryID);
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) { printf("no default Metal device\n"); return 1; }
    printf("default=%s\n", d.name.UTF8String);
    printf("lowPower=%d headless=%d removable=%d unifiedMemory=%d\n", d.lowPower, d.headless, d.removable, d.hasUnifiedMemory);
    printf("recommendedMaxWorkingSetSize=%llu MB maxBufferLength=%lu MB maxThreadgroupMemoryLength=%lu\n",
           d.recommendedMaxWorkingSetSize >> 20, (unsigned long)(d.maxBufferLength >> 20), (unsigned long)d.maxThreadgroupMemoryLength);
    printf("maxThreadsPerThreadgroup=%lux%lux%lu\n", (unsigned long)d.maxThreadsPerThreadgroup.width,
           (unsigned long)d.maxThreadsPerThreadgroup.height, (unsigned long)d.maxThreadsPerThreadgroup.depth);
    yn("supportsFamily Apple4", [d supportsFamily:MTLGPUFamilyApple4]);
    yn("supportsFamily Apple5", [d supportsFamily:MTLGPUFamilyApple5]);
    yn("supportsFamily Apple6", [d supportsFamily:MTLGPUFamilyApple6]);
    yn("supportsFamily Apple7", [d supportsFamily:MTLGPUFamilyApple7]);
    yn("supportsFamily Apple8", [d supportsFamily:MTLGPUFamilyApple8]);
    yn("supportsFamily Apple9", [d supportsFamily:MTLGPUFamilyApple9]);
    yn("supportsFamily Mac2", [d supportsFamily:MTLGPUFamilyMac2]);
    yn("supportsFamily Common3", [d supportsFamily:MTLGPUFamilyCommon3]);
    yn("supportsFamily Metal3", [d supportsFamily:MTLGPUFamilyMetal3]);
    if (@available(macOS 26.0, *)) yn("supportsFamily Metal4", [d supportsFamily:MTLGPUFamilyMetal4]);
    printf("argumentBuffersSupport=%ld (0=Tier1 1=Tier2)\n", (long)d.argumentBuffersSupport);
    yn("supportsRasterizationRateMapWithLayerCount:2", [d supportsRasterizationRateMapWithLayerCount:2]);
    yn("supportsBCTextureCompression", d.supportsBCTextureCompression);
    yn("supports32BitFloatFiltering", d.supports32BitFloatFiltering);
    yn("supports32BitMSAA", d.supports32BitMSAA);
    yn("supportsPullModelInterpolation", d.supportsPullModelInterpolation);
    yn("supportsShaderBarycentricCoordinates", d.supportsShaderBarycentricCoordinates);
    yn("supportsFunctionPointers", d.supportsFunctionPointers);
    yn("supportsRaytracing", d.supportsRaytracing);
    yn("supportsDynamicLibraries", d.supportsDynamicLibraries);
    for (int c : {1, 2, 4, 8}) {
      char b[64]; snprintf(b, sizeof b, "supportsTextureSampleCount %d", c);
      yn(b, [d supportsTextureSampleCount:c]);
    }
    // ASTC: the Apple GPUs sample it; on Macs the property below is the answer (no public query: try a texture).
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_4x4_LDR width:16 height:16 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> astc = [d newTextureWithDescriptor:td];
    yn("ASTC 4x4 LDR texture creatable", astc != nil);
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBC1_RGBA width:16 height:16 mipmapped:NO];
    id<MTLTexture> bc = [d newTextureWithDescriptor:td];
    yn("BC1 texture creatable", bc != nil);
    // Memoryless (tile memory render targets): Apple GPUs only.
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeMemoryless;
    id<MTLTexture> ml = nil;
    @try { ml = [d newTextureWithDescriptor:td]; } @catch (NSException* e) { printf("memoryless: exception %s\n", e.reason.UTF8String); }
    yn("Memoryless texture creatable", ml != nil);
    yn("supportsFamily Apple1", [d supportsFamily:MTLGPUFamilyApple1]);
    NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
    printf("macOS %ld.%ld.%ld\n", (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion);
  }
  return 0;
}
