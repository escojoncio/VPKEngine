// VPKTAA: VPKEngine's temporal antialiasing for the headset, one full-screen pass per eye (per view).
//
// It replaces the scene's resolve: it reads the frame rendered with a sub-pixel jitter (possibly at
// a lower internal resolution) and the eye's history, and writes the antialiased colour at the
// output size plus the next history. For each output pixel:
//   1. reconstruct this frame at the pixel centre from the 3x3 render samples around it (a Gaussian
//      fit, sharpened against the 3x3 mean the way contrast-adaptive sharpening does, never past the
//      samples' own range: no halos);
//   2. reproject the history with the motion of the nearest surface in the 3x3 (an edge moves with
//      what is in front of it). A translated PS4 game gives no motion vectors, so motion is the
//      camera's: the depth and one matrix from this frame's clip space to the previous frame's
//      (built from the head pose and the eye's projection by the presenter). An optional motion
//      texture overrides it where the client has object motion;
//   3. reject what the history cannot hold: a surface that was hidden in the previous frame (the
//      history keeps each pixel's view distance in alpha), colour outside the neighbourhood's
//      variance box (narrower at depth edges), and what the reactive mask covers (blended draws:
//      particles, transparencies);
//   4. blend: more history while still, less while moving or where history and current disagree.
// Colour is handled in a reversible tonemapped space (each sample weighed by 1 / (1 + brightness),
// so a bright spot does not smear) and in YCoCg for the variance box.
// Cost: the 3x3's depths come from 3 gathers + 1 read, the reconstruction weight is separable
// (6 exps) and the nearest sample is found per axis; empty samples (sky, cleared depth) keep a
// finite distance in the history (60000) and skip the disocclusion test.
//
// Bindings (fragment stage):
//   texture 0  this frame's colour, render size, linear HDR (any float format)
//   texture 1  this frame's depth, render size (reverse Z when VPKTAA_REVERSE_Z is set in flags)
//   texture 2  the history to read, output size, RGBA16F: rgb plain HDR colour, a view distance
//   texture 3  optional reactive mask, render size, r in [0, 1] (bound or not: flags)
//   texture 4  optional motion, render size, rg = previous uv - current uv (bound or not: flags)
//   buffer  0  VpktaaParams
// Outputs: [[color(0)]] the antialiased colour, [[color(1)]] the next history (same format as
// texture 2). Vertex stage: vpktaa_vertex draws one triangle covering the target (3 vertices).
#include <metal_stdlib>
using namespace metal;

enum : uint {
    VPKTAA_HISTORY_VALID = 1u << 0,  // texture 2 holds this view's previous output
    VPKTAA_PERIPHERY = 1u << 1,      // a wide, low-detail view: no sharpening
    VPKTAA_REVERSE_Z = 1u << 2,      // depth 1 near, 0 far (else 0 near, 1 far)
    VPKTAA_REACTIVE = 1u << 3,       // texture 3 is bound
    VPKTAA_MOTION = 1u << 4,         // texture 4 is bound
};

struct VpktaaParams {
    float4x4 clipToPrevClip;  // this frame's unjittered clip space -> the previous frame's
    float2 jitter;            // this frame's jitter, in render uv
    float2 renderSize;        // render target size in pixels
    float2 outputSize;        // output (and history) size in pixels
    float2 viewZ;             // view distance = 1 / (viewZ.x * depth + viewZ.y)
    float sharpness;          // 0..1
    uint flags;
};

// Tunables (history weight still / at 8 px per frame, variance box width at flat areas / depth
// edges, disocclusion tolerance relative to the distance).
constant float kKeepStill = 0.93f;
constant float kKeepMoving = 0.80f;
constant float kKeepMax = 0.97f;
constant float kSigmaFlat = 1.25f;
constant float kSigmaEdge = 0.85f;
constant float kDisocclusion = 0.05f;
constant float kSkyDistance = 60000.0f;  // history distance where nothing was drawn (fits half)

struct VpktaaVaryings {
    float4 position [[position]];
};

vertex VpktaaVaryings vpktaa_vertex(uint id [[vertex_id]]) {
    // One triangle over the whole target: (-1,-1) (3,-1) (-1,3).
    float2 p = float2((id << 1) & 2, id & 2);
    VpktaaVaryings out;
    out.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}

struct VpktaaOut {
    float4 color [[color(0)]];
    float4 history [[color(1)]];
};

static inline float maxOf(float3 c) { return max(c.r, max(c.g, c.b)); }

// Reversible tonemap.
static inline float3 compress(float3 c) { return c / (1.0f + maxOf(c)); }
static inline float3 expand(float3 c) {
    float m = maxOf(c);
    if (m > 0.9999f) c *= 0.9999f / m;
    return c / max(1.0f - maxOf(c), 1.0e-4f);
}

// NaN/Inf would live in the history forever.
static inline float3 sane(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0.0f) : clamp(c, float3(0.0f), float3(60000.0f));
}

static inline float3 toYCoCg(float3 c) {
    return float3(0.25f * c.r + 0.5f * c.g + 0.25f * c.b, 0.5f * c.r - 0.5f * c.b, -0.25f * c.r + 0.5f * c.g - 0.25f * c.b);
}
static inline float3 fromYCoCg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// Nothing drawn there: the depth clear value (far plane).
static inline bool isEmpty(float depth, bool reverseZ) { return reverseZ ? depth <= 0.0f : depth >= 1.0f; }

static inline float viewDistance(float depth, constant VpktaaParams& p) {
    return min(1.0f / max(p.viewZ.x * depth + p.viewZ.y, 1.0e-6f), kSkyDistance);
}

// Index (0..2) of the smallest of three squared distances (the first on a tie).
static inline int nearestOf(float3 d) {
    return d.y < d.x ? (d.z < d.y ? 2 : 1) : (d.z < d.x ? 2 : 0);
}

static inline float3 historyTap(texture2d<float> history, sampler s, float2 px, float2 size) {
    return history.sample(s, clamp(px, float2(0.5f), size - 0.5f) / size).rgb;
}

// History with a 5-tap Catmull-Rom (the four corner taps dropped): sharper than bilinear, so the
// history does not soften a little more every frame. `pixel` in output pixels.
static float3 historyCatmullRom(texture2d<float> history, sampler s, float2 pixel, float2 size) {
    float2 position = pixel - 0.5f;
    float2 centre = floor(position) + 0.5f;
    float2 f = position - floor(position);
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);
    float2 w12 = w1 + w2;
    float2 p0 = centre - 1.0f;
    float2 p3 = centre + 2.0f;
    float2 p12 = centre + w2 / w12;
    float a = w12.x * w0.y, b = w0.x * w12.y, c = w12.x * w12.y, d = w3.x * w12.y, e = w12.x * w3.y;
    float3 sum = historyTap(history, s, float2(p12.x, p0.y), size) * a + historyTap(history, s, float2(p0.x, p12.y), size) * b +
                 historyTap(history, s, p12, size) * c + historyTap(history, s, float2(p3.x, p12.y), size) * d +
                 historyTap(history, s, float2(p12.x, p3.y), size) * e;
    return compress(sane(max(sum / max(a + b + c + d + e, 1.0e-4f), float3(0.0f))));
}

// Clip toward the box's centre (keeps the history's hue where a per-channel clamp turns blocky).
static inline float3 clipToBox(float3 h, float3 lo, float3 hi) {
    float3 centre = 0.5f * (hi + lo);
    float3 extent = max(0.5f * (hi - lo), float3(1.0e-5f));
    float3 offset = h - centre;
    float3 units = abs(offset / extent);
    float largest = max(units.x, max(units.y, units.z));
    return largest > 1.0f ? centre + offset / largest : h;
}

fragment VpktaaOut vpktaa_fragment(VpktaaVaryings in [[stage_in]],
                                   texture2d<float> current [[texture(0)]],
                                   depth2d<float> depthTex [[texture(1)]],
                                   texture2d<float> history [[texture(2)]],
                                   texture2d<float> reactiveTex [[texture(3)]],
                                   texture2d<float> motionTex [[texture(4)]],
                                   constant VpktaaParams& p [[buffer(0)]]) {
    constexpr sampler linearClamp(filter::linear, address::clamp_to_edge);
    constexpr sampler pointClamp(filter::nearest, address::clamp_to_edge);
    const int2 renderMax = int2(p.renderSize) - 1;
    const float2 uv = in.position.xy / p.outputSize;
    const float2 ratio = p.renderSize / p.outputSize;  // render pixels per output pixel
    // A render sample k sits at (k + 0.5 - jitter) in unjittered render pixels.
    const float2 renderPoint = uv * p.renderSize + p.jitter * p.renderSize;
    const int2 base = int2(floor(renderPoint));
    const bool reverseZ = (p.flags & VPKTAA_REVERSE_Z) != 0;

    // The 3x3's depths, row-major from (base - 1): a gather at the corner between texels (x-1, y-1)
    // and (x, y) returns (x-1,y) (x,y) (x,y-1) (x-1,y-1); clamp to edge matches the reads' clamp.
    float depths[9];
    {
        const float2 corner = float2(base) / float2(depthTex.get_width(), depthTex.get_height());
        const float4 g0 = depthTex.gather(pointClamp, corner);
        const float4 g1 = depthTex.gather(pointClamp, corner, int2(1, 0));
        const float4 g2 = depthTex.gather(pointClamp, corner, int2(0, 1));
        depths[0] = g0.w; depths[1] = g0.z; depths[2] = g1.z;
        depths[3] = g0.x; depths[4] = g0.y; depths[5] = g1.y;
        depths[6] = g2.x; depths[7] = g2.y;
        depths[8] = depthTex.read(uint2(clamp(base + 1, int2(0), renderMax)));
    }

    // Squared distances of the sample columns / rows to this pixel's centre, in output pixels: the
    // Gaussian (Blackman-Harris fit) is separable, and the nearest sample is the nearest column
    // with the nearest row.
    const float2 offset = float2(base) + 0.5f - renderPoint;
    float3 dx = (float3(-1.0f, 0.0f, 1.0f) + offset.x) / ratio.x;
    float3 dy = (float3(-1.0f, 0.0f, 1.0f) + offset.y) / ratio.y;
    dx *= dx;
    dy *= dy;
    const float3 wx = exp(-2.29f * dx), wy = exp(-2.29f * dy);
    const int nearestIndex = nearestOf(dy) * 3 + nearestOf(dx);

    float3 m1 = 0.0f, m2 = 0.0f, gauss = 0.0f, boxLo = float3(1.0e9f), boxHi = float3(-1.0e9f);
    float ownDepth = 0.0f, ownWeight = 0.0f;
    float nearestDepth = depths[4], farthestDepth = depths[4];
    int nearestSurface = 4;
    for (int i = 0; i < 9; ++i) {
        const int2 o = int2(i % 3, i / 3);
        const float3 s = toYCoCg(compress(sane(current.read(uint2(clamp(base + o - 1, int2(0), renderMax))).rgb)));
        const float w = wx[o.x] * wy[o.y];
        m1 += s;
        m2 += s * s;
        gauss += s * w;
        boxLo = min(boxLo, s);
        boxHi = max(boxHi, s);
        const float z = depths[i];
        if (i == nearestIndex) ownDepth = z, ownWeight = w;
        if (reverseZ ? z > nearestDepth : z < nearestDepth) nearestDepth = z, nearestSurface = i;
        farthestDepth = reverseZ ? min(farthestDepth, z) : max(farthestDepth, z);
    }
    const int2 surfacePixel = clamp(base + int2(nearestSurface % 3, nearestSurface / 3) - 1, int2(0), renderMax);
    const int2 ownPixel = clamp(base + int2(nearestIndex % 3, nearestIndex / 3) - 1, int2(0), renderMax);

    // Motion of the nearest surface: the client's if it has one, else the camera's from depth.
    float2 motion = 0.0f;
    float expectedDistance = 0.0f;  // the surface's distance in the previous frame (0: unknown)
    bool haveMotion = false;
    if (p.flags & VPKTAA_MOTION) {
        const float2 mv = motionTex.read(uint2(surfacePixel)).rg;
        if (all(isfinite(mv)) && all(abs(mv) < 1.0e3f)) motion = mv, haveMotion = true;
    }
    {
        const float2 suv = (float2(surfacePixel) + 0.5f) / p.renderSize - p.jitter;  // unjittered uv
        const float2 ndc = float2(suv.x * 2.0f - 1.0f, 1.0f - suv.y * 2.0f);
        // (ndc, depth, 1) is this frame's clip position divided by its w, so the previous clip
        // position comes out divided by it too. Nothing drawn (the clear value): the far plane, a
        // direction with an infinite one; it moves with the camera but holds no distance to test.
        const bool empty = isEmpty(nearestDepth, reverseZ);
        const float4 prev = p.clipToPrevClip * float4(ndc, nearestDepth, 1.0f);
        if (prev.w > 1.0e-6f) {
            if (!empty) expectedDistance = prev.w * viewDistance(nearestDepth, p);
            if (!haveMotion) {
                const float2 prevNdc = prev.xy / prev.w;
                motion = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f) - suv;
            }
        }
        if (!all(isfinite(motion))) motion = 0.0f;
    }
    const float speed = min(length(motion * p.outputSize), 64.0f);  // output pixels per frame
    const float still = 1.0f - saturate(speed * 0.5f);

    // This frame at the pixel, sharpened against the 3x3 mean (the accumulation converges on the
    // sharpened value: no separate pass, and the history is never sharpened twice).
    gauss /= max(dot(wx, float3(1.0f)) * dot(wy, float3(1.0f)), 1.0e-5f);
    const float3 mean = m1 / 9.0f;
    float3 current0 = gauss;
    if (!(p.flags & VPKTAA_PERIPHERY)) {
        // Contrast-adaptive: less where the neighbourhood already spans a wide range or sits near
        // black/white; more while moving, where the history blurs most. Never past the samples.
        const float amp = sqrt(saturate(min(boxLo.x, 1.0f - boxHi.x) / max(boxHi.x, 1.0e-4f)));
        const float amount = p.sharpness * amp * (1.0f + 0.6f * (1.0f - still));
        current0 = clamp(gauss + amount * 1.2f * (gauss - mean), boxLo, boxHi);
    }

    // Variance box: narrow at depth edges (the history cannot drag an edge's old place along),
    // wide on flat areas (less flicker of fine detail).
    const float nearZ = viewDistance(nearestDepth, p), farZ = viewDistance(farthestDepth, p);
    const float depthRange = (farZ - nearZ) / max(farZ, 1.0e-6f);
    const float edge = smoothstep(0.02f, 0.15f, depthRange);
    const float3 sigma = sqrt(max(m2 / 9.0f - mean * mean, float3(0.0f)));
    const float k = mix(kSigmaFlat, kSigmaEdge, edge);
    const float3 lo = min(max(mean - k * sigma, boxLo), current0);
    const float3 hi = max(min(mean + k * sigma, boxHi), current0);

    float reactive = 0.0f;
    if (p.flags & VPKTAA_REACTIVE) reactive = saturate(reactiveTex.read(uint2(ownPixel)).r);

    const float ownDistance = isEmpty(ownDepth, reverseZ) ? kSkyDistance : viewDistance(ownDepth, p);
    float3 result = current0;
    const float2 prevUv = uv + motion;
    const bool inside = all(prevUv >= 0.0f) && all(prevUv <= 1.0f);
    if ((p.flags & VPKTAA_HISTORY_VALID) && inside) {
        const float2 prevPixel = prevUv * p.outputSize;
        const float3 h = clipToBox(toYCoCg(historyCatmullRom(history, linearClamp, prevPixel, p.outputSize)), lo, hi);
        float keep = mix(kKeepStill, kKeepMoving, saturate(speed * 0.125f));
        // Anti-flicker: where history and current agree keep more, where they disagree take more
        // of the current.
        const float agree = 1.0f - abs(current0.x - h.x) / max(max(current0.x, h.x), 0.2f);
        keep = mix(keep - 0.12f, keep, agree * agree);
        // Disocclusion: the history held something nearer there (this surface was hidden).
        if (expectedDistance > 0.0f) {
            const float held = history.read(uint2(clamp(prevPixel, float2(0.0f), p.outputSize - 1.0f))).a;
            const float nearer = (expectedDistance - held) / expectedDistance;
            const float tolerance = max(kDisocclusion, 1.5f * depthRange);
            keep *= 1.0f - smoothstep(tolerance, 2.0f * tolerance, nearer);
        }
        keep *= 1.0f - 0.75f * reactive;
        // Below 100 %: where no sample falls near this pixel the history knows it better.
        if (ratio.x < 0.999f) keep = 1.0f - (1.0f - keep) * mix(0.35f, 1.0f, ownWeight);
        result = mix(current0, h, clamp(keep, 0.0f, kKeepMax));
    }
    const float3 color = expand(max(fromYCoCg(result), float3(0.0f)));
    VpktaaOut out;
    out.color = float4(color, 1.0f);
    out.history = float4(color, ownDistance);
    return out;
}
