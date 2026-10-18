// VPKTAA nativo (VPVulkan): TAA temporal como pase interno del motor, ambos ojos en un dispatch
// (z = ojo, capas de un array: multiview). Matemática idéntica a gpu/vulkan/src/taa_native.cpp
// (probada en Linux). Dos kernels:
//   vpktaa_native_resolve  color con jitter (+ profundidad del juego, MSAA o no, + vectores de
//                          SpaceWarp opcionales) + historia -> historia nueva (rgb lineal, a =
//                          distancia de vista); sin afilar: la historia nunca se afila dos veces.
//   vpktaa_native_rcas     RCAS (FSR 1) sobre la historia nueva -> salida del ojo (codifica sRGB si
//                          VPKN_OUT_SRGB: la salida es una vista unorm de una imagen sRGB).
//   vpktaa_native_depth_*  profundidad MSAA -> depth32Float con la muestra más cercana (o la 0),
//                          para consumidores de profundidad de 1 muestra (PTW del compositor).
// Pipeline por píxel g (texels del color):
//   1. reconstrucción en el centro sin jitter: 3x3 alrededor de g + base con pesos gaussianos
//      precalculados en CPU (el desplazamiento es uniforme), en espacio tonemapeado reversible y YCoCg;
//   2. profundidad: la más cercana del 3x3 (dilatación, un borde se mueve con lo que tiene delante);
//   3. movimiento: objeto (SpaceWarp, si lo hay) y luego cámara por reproyección desde profundidad
//      + pose (rotación y traslación; sin profundidad o cielo: solo rotación);
//   4. historia Catmull-Rom de 5 lecturas, recortada contra la caja de varianza YCoCg (estrecha en
//      bordes de profundidad), peso adaptativo por velocidad, acuerdo y desoclusión por distancia.
//   5. máscara de UI (R8, la escribe el motor en los draws de UI/2D del pase de ojo; ver
//      taa_native.h): donde vale m > 0 la base de la reconstrucción se acerca al texel actual SIN
//      jitter (rec += (cur - rec)·m), el peso de la historia cae a keep·(1-m) (con m ~ 1 ni se lee la
//      historia: sin clamp) y RCAS no afila (c = mix(c, e, m)). Réplica de taa::maskedBase /
//      maskedHistoryWeight (probadas en Linux).
#include <metal_stdlib>
using namespace metal;

enum : uint {
    VPKN_HISTORY = 1u << 0,   // la historia de este ojo es válida
    VPKN_DEPTH = 1u << 1,     // profundidad ligada
    VPKN_REVZ = 1u << 2,      // profundidad invertida (1 cerca)
    VPKN_MOTION = 1u << 3,    // vectores de movimiento (SpaceWarp) ligados
    VPKN_OUT_SRGB = 1u << 4,  // codificar sRGB al escribir la salida
    VPKN_MASK = 1u << 5,      // máscara de UI ligada (texture(2))
};

struct VpknEye {
    float4 rot0, rot1, rot2;  // ojo actual -> ojo anterior (filas)
    float4 trans;             // xyz
    float4 tanCur, tanPrev;   // l r t b (sin jitter)
    float4 crop;              // xy tamaño, zw origen (fracciones del color)
    float4 depthAB;           // texel de profundidad = texel de color * xy + zw
    float4 mvAB;              // texel de movimiento = texel de color * xy + zw
    uint4 slices;             // color, profundidad, movimiento, flags
    float4 maskAB;            // texel de la máscara = texel de color * xy + zw
    uint4 maskInfo;           // x: capa de la máscara
};

struct VpknParams {
    VpknEye eye[2];
    float4 w0, w1;            // pesos 0..7 del 3x3 (fila mayor)
    float4 w2;                // x: peso 8
    int4 base;                // xy: base de reconstrucción, zw: offset del texel más cercano
    float4 depthP;            // near, far (0 = inf), distancia cielo, tolerancia de desoclusión
    float4 keep;              // quieto, moviéndose, máximo, nitidez RCAS
    float4 mvP;               // escala xy (texel -> uv del recorte, actual - anterior)
    uint4 size;               // ancho, alto del color; muestras de profundidad; primer ojo del dispatch
};

static inline float maxOf(float3 c) { return max(c.r, max(c.g, c.b)); }
static inline float3 compress(float3 c) { return c / (1.0f + maxOf(c)); }
static inline float3 expand(float3 c) {
    float m = maxOf(c);
    if (m > 0.9999f) c *= 0.9999f / m;
    return c / max(1.0f - maxOf(c), 1.0e-4f);
}
static inline float3 sane(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0.0f) : clamp(c, float3(0.0f), float3(60000.0f));
}
static inline float3 toYCoCg(float3 c) {
    return float3(0.25f * c.r + 0.5f * c.g + 0.25f * c.b, 0.5f * c.r - 0.5f * c.b, -0.25f * c.r + 0.5f * c.g - 0.25f * c.b);
}
static inline float3 fromYCoCg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// taa_native.cpp viewDistance.
static inline float viewDistance(float d, constant VpknParams& p, bool rev) {
    d = rev ? 1.0f - d : d;
    const float n = max(p.depthP.x, 1.0e-4f), f = p.depthP.y;
    const float z = f <= 0.0f ? n / max(1.0f - d, 1.0e-7f) : f * n / max(f - d * (f - n), 1.0e-7f);
    return min(z, p.depthP.z);
}
static inline bool emptyDepth(float d, bool rev) { return rev ? d <= 0.0f : d >= 1.0f; }
static inline bool nearer(float a, float b, bool rev) { return rev ? a > b : a < b; }

// taa_native.cpp reprojectUv: uv del ojo (fila 0 arriba) a distancia `dist` (0 = solo rotación).
static inline bool reproject(constant VpknEye& e, float2 uv, float dist, thread float2& prevUv, thread float& prevDist) {
    const float3 d = float3(-e.tanCur.x + uv.x * (e.tanCur.x + e.tanCur.y), e.tanCur.z - uv.y * (e.tanCur.z + e.tanCur.w), -1.0f);
    float3 q = float3(dot(e.rot0.xyz, d), dot(e.rot1.xyz, d), dot(e.rot2.xyz, d));
    if (dist > 0.0f) q = q * dist + e.trans.xyz;
    if (q.z >= -1.0e-6f) return false;
    const float z = -q.z;
    prevDist = dist > 0.0f ? z : 0.0f;
    prevUv = float2((q.x / z + e.tanPrev.x) / (e.tanPrev.x + e.tanPrev.y), (e.tanPrev.z - q.y / z) / (e.tanPrev.z + e.tanPrev.w));
    return true;
}

// Cobertura de UI en el texel de color `t` (0 si no hay máscara; el ruido de R8 cuenta como 0).
static inline float uiCoverage(texture2d_array<float> mask, constant VpknEye& e, int2 t) {
    if (!(e.slices.w & VPKN_MASK)) return 0.0f;
    const uint2 q = uint2(clamp(int2(floor(float2(t) * e.maskAB.xy + e.maskAB.zw)), int2(0),
                                int2(mask.get_width(), mask.get_height()) - 1));
    const float m = mask.read(q, e.maskInfo.x).r;
    return m <= 1.0f / 255.0f ? 0.0f : min(m, 1.0f);
}

static inline float3 histTap(texture2d_array<float> h, sampler s, float2 px, float2 size, uint slice) {
    return h.sample(s, clamp(px, float2(0.5f), size - 0.5f) / size, slice).rgb;
}
// Catmull-Rom de 5 lecturas bilineales (taa_native.cpp catmullRom5). `px` en texels.
static float3 historyCR(texture2d_array<float> h, sampler s, float2 px, float2 size, uint slice) {
    const float2 pos = px - 0.5f, c = floor(pos) + 0.5f, f = pos - floor(pos);
    const float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    const float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    const float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    const float2 w3 = f * f * (-0.5f + 0.5f * f);
    const float2 w12 = w1 + w2, p0 = c - 1.0f, p3 = c + 2.0f, p12 = c + w2 / w12;
    const float a = w12.x * w0.y, b = w0.x * w12.y, cc = w12.x * w12.y, d = w3.x * w12.y, e = w12.x * w3.y;
    const float3 sum = histTap(h, s, float2(p12.x, p0.y), size, slice) * a + histTap(h, s, float2(p0.x, p12.y), size, slice) * b +
                       histTap(h, s, p12, size, slice) * cc + histTap(h, s, float2(p3.x, p12.y), size, slice) * d +
                       histTap(h, s, float2(p12.x, p3.y), size, slice) * e;
    return compress(sane(max(sum / max(a + b + cc + d + e, 1.0e-4f), float3(0.0f))));
}

static inline float3 clipToBox(float3 h, float3 lo, float3 hi) {
    const float3 c = 0.5f * (hi + lo), ext = max(0.5f * (hi - lo), float3(1.0e-5f)), o = h - c;
    const float3 u = abs(o / ext);
    const float l = max(u.x, max(u.y, u.z));
    return l > 1.0f ? c + o / l : h;
}

// Dos variantes compiladas por macro (taa_native_metal.mm): VPKN_DEPTH_MS = 1 lee la profundidad
// MSAA del juego muestra a muestra (la más cercana), sin resolverla antes.
#ifndef VPKN_DEPTH_MS
#define VPKN_DEPTH_MS 0
#endif
#if VPKN_DEPTH_MS
typedef depth2d_ms_array<float> VpknDepthTex;
static inline float depthAt(VpknDepthTex d, int2 t, uint slice, uint samples, bool rev) {
    const uint2 q = uint2(clamp(t, int2(0), int2(d.get_width(), d.get_height()) - 1));
    float z = d.read(q, slice, 0);
    for (uint s = 1; s < samples; ++s) {
        const float v = d.read(q, slice, s);
        z = nearer(v, z, rev) ? v : z;
    }
    return z;
}
#else
typedef depth2d_array<float> VpknDepthTex;
static inline float depthAt(VpknDepthTex d, int2 t, uint slice, uint, bool) {
    const uint2 q = uint2(clamp(t, int2(0), int2(d.get_width(), d.get_height()) - 1));
    return d.read(q, slice);
}
#endif

kernel void vpktaa_native_resolve(texture2d_array<float, access::read> color [[texture(0)]],
                                  VpknDepthTex depth [[texture(1)]],
                                  texture2d_array<float> uiMask [[texture(2)]],
                                  texture2d_array<float> motion [[texture(3)]],
                                  texture2d_array<float> history [[texture(4)]],
                                  texture2d_array<float, access::write> historyOut [[texture(5)]],
                                  constant VpknParams& p [[buffer(0)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    const uint W = p.size.x, H = p.size.y;
    if (gid.x >= W || gid.y >= H || gid.z + p.size.w > 1) return;
    const uint eyeIx = gid.z + p.size.w;
    constant VpknEye& e = p.eye[eyeIx];
    const uint flags = e.slices.w;
    const bool rev = (flags & VPKN_REVZ) != 0;
    const int2 g = int2(gid.xy);
    const float2 size = float2(W, H);
    const float2 cropSize = e.crop.xy * size, cropOrg = e.crop.zw * size;
    const float2 pc = float2(g) + 0.5f;
    const bool inside = all(pc >= cropOrg) && all(pc <= cropOrg + cropSize);
    const int2 maxT = int2(W, H) - 1;
    if (!inside) {
        historyOut.write(float4(sane(color.read(uint2(g), e.slices.x).rgb), p.depthP.z), uint2(g), eyeIx);
        return;
    }

    // 1. Reconstrucción + momentos.
    const float wts[9] = {p.w0.x, p.w0.y, p.w0.z, p.w0.w, p.w1.x, p.w1.y, p.w1.z, p.w1.w, p.w2.x};
    const int2 b = g + p.base.xy;
    float3 rec = 0.0f, m1 = 0.0f, m2 = 0.0f, lo = float3(1.0e9f), hi = float3(-1.0e9f);
    for (int i = 0; i < 9; ++i) {
        const int2 t = clamp(b + int2(i % 3, i / 3) - 1, int2(0), maxT);
        const float3 s = toYCoCg(compress(sane(color.read(uint2(t), e.slices.x).rgb)));
        rec += s * wts[i];
        m1 += s;
        m2 += s * s;
        lo = min(lo, s);
        hi = max(hi, s);
    }

    // Máscara de UI: el texel actual SIN reconstruir (la UI no lleva jitter) sustituye a la base.
    const float uiM = uiCoverage(uiMask, e, g);
    if (uiM > 0.0f) rec = mix(rec, toYCoCg(compress(sane(color.read(uint2(g), e.slices.x).rgb))), uiM);

    // 2. Profundidad: más cercana y más lejana del 3x3, y la propia (texel más cercano).
    float zNear = 0.0f, zFar = 0.0f, zOwn = 0.0f;
    int2 nearT = b + p.base.zw;
    const bool haveDepth = (flags & VPKN_DEPTH) != 0;
    if (haveDepth) {
        const int2 dc = int2(floor(float2(b) * e.depthAB.xy + e.depthAB.zw));
        zOwn = depthAt(depth, int2(floor(float2(b + p.base.zw) * e.depthAB.xy + e.depthAB.zw)), e.slices.y, p.size.z, rev);
        zNear = zFar = zOwn;
        for (int i = 0; i < 9; ++i) {
            const int2 o = int2(i % 3, i / 3) - 1;
            const float z = depthAt(depth, dc + o, e.slices.y, p.size.z, rev);
            if (nearer(z, zNear, rev)) zNear = z, nearT = b + o;
            if (nearer(zFar, z, rev)) zFar = z;
        }
    }
    const bool sky = !haveDepth || emptyDepth(zNear, rev);
    const float distNear = sky ? 0.0f : viewDistance(zNear, p, rev);
    const float distOwn = (!haveDepth || emptyDepth(zOwn, rev)) ? p.depthP.z : viewDistance(zOwn, p, rev);

    // 3. Movimiento.
    float2 uv = (pc - cropOrg) / cropSize;
    float2 uvObj = uv;
    if (flags & VPKN_MOTION) {
        const uint2 mt = uint2(clamp(int2(floor(float2(nearT) * e.mvAB.xy + e.mvAB.zw)), int2(0),
                                     int2(motion.get_width(), motion.get_height()) - 1));
        const float2 mv = motion.read(mt, e.slices.z).rg;
        if (all(isfinite(mv)) && all(abs(mv) < 1.0e3f)) uvObj -= mv * p.mvP.xy;
    }
    float2 prevUv = uvObj;
    float expected = 0.0f;
    const bool onScreen = reproject(e, uvObj, distNear, prevUv, expected);
    const float2 prevPx = cropOrg + prevUv * cropSize;
    const float speed = min(length(prevPx - pc), 64.0f);

    // 4. Historia.
    const float3 mean = m1 / 9.0f;
    const float3 sigma = sqrt(max(m2 / 9.0f - mean * mean, float3(0.0f)));
    float edge = 0.0f;
    if (haveDepth && !sky) {
        const float dn = viewDistance(zNear, p, rev), df = viewDistance(zFar, p, rev);
        edge = smoothstep(0.02f, 0.15f, (df - dn) / max(df, 1.0e-6f));
    }
    const float k = mix(1.25f, 0.85f, edge);
    const float3 bLo = min(max(mean - k * sigma, lo), rec), bHi = max(min(mean + k * sigma, hi), rec);
    float3 result = rec;
    const bool prevIn = onScreen && all(prevPx >= cropOrg) && all(prevPx <= cropOrg + cropSize);
    if ((flags & VPKN_HISTORY) && prevIn && uiM < 0.999f) {
        constexpr sampler lin(filter::linear, address::clamp_to_edge);
        const float3 h = clipToBox(toYCoCg(historyCR(history, lin, prevPx, size, eyeIx)), bLo, bHi);
        float keep = mix(p.keep.x, p.keep.y, saturate(speed * 0.125f));
        const float agree = 1.0f - abs(rec.x - h.x) / max(max(rec.x, h.x), 0.2f);
        keep = mix(keep - 0.12f, keep, agree * agree) * (1.0f - uiM);
        if (expected > 0.0f) {
            const float held = history.read(uint2(clamp(prevPx, float2(0.0f), size - 1.0f)), eyeIx).a;
            const float hid = (expected - held) / expected;  // > 0: antes había algo más cerca
            keep *= 1.0f - smoothstep(p.depthP.w, 2.0f * p.depthP.w, hid);
        }
        result = mix(rec, h, clamp(keep, 0.0f, p.keep.z));
    }
    historyOut.write(float4(expand(max(fromYCoCg(result), float3(0.0f))), distOwn), uint2(g), eyeIx);
}

static inline float3 srgbEncode(float3 c) {
    c = saturate(c);
    return select(1.055f * pow(c, 1.0f / 2.4f) - 0.055f, c * 12.92f, c <= 0.0031308f);
}

static inline float3 rcasTap(texture2d_array<float, access::read> t, int2 q, int2 m, uint slice) {
    return saturate(t.read(uint2(clamp(q, int2(0), m)), slice).rgb);
}

// RCAS (AMD FidelityFX FSR 1, MIT): afilado robusto por contraste, sin halos, en la cruz de 5 texels.
kernel void vpktaa_native_rcas(texture2d_array<float, access::read> src [[texture(0)]],
                               texture2d_array<float, access::write> out [[texture(1)]],
                               texture2d_array<float> uiMask [[texture(2)]],
                               constant VpknParams& p [[buffer(0)]],
                               uint3 gid [[thread_position_in_grid]]) {
    const uint W = p.size.x, H = p.size.y;
    if (gid.x >= W || gid.y >= H || gid.z + p.size.w > 1) return;
    const uint eyeIx = gid.z + p.size.w;
    const int2 g = int2(gid.xy), m = int2(W, H) - 1;
    const float3 bN = rcasTap(src, g + int2(0, -1), m, eyeIx), dW = rcasTap(src, g + int2(-1, 0), m, eyeIx);
    const float3 e = rcasTap(src, g, m, eyeIx), fE = rcasTap(src, g + int2(1, 0), m, eyeIx);
    const float3 hS = rcasTap(src, g + int2(0, 1), m, eyeIx);
    float3 c = e;
    const float sharp = p.keep.w;
    const float uiM = uiCoverage(uiMask, p.eye[eyeIx], g);  // UI: sin afilado
    if (sharp > 0.0f && uiM < 0.999f) {
        const float3 mn = min(min(bN, dW), min(fE, hS)), mx = max(max(bN, dW), max(fE, hS));
        const float3 hitMin = min(mn, e) / (4.0f * max(mx, float3(1.0e-4f)));
        const float3 hitMax = (1.0f - max(mx, e)) / min(4.0f * mn - 4.0f, float3(-1.0e-4f));
        const float3 l = max(-hitMin, hitMax);
        const float lobe = max(-0.1875f, min(max(l.r, max(l.g, l.b)), 0.0f)) * sharp;
        c = mix(saturate((lobe * (bN + dW + fE + hS) + e) / (4.0f * lobe + 1.0f)), e, uiM);
    }
    if (p.eye[eyeIx].slices.w & VPKN_OUT_SRGB) c = srgbEncode(c);
    out.write(float4(c, 1.0f), uint2(g), eyeIx);
}

// Profundidad MSAA -> depth32Float de 1 muestra (render: los kernels no escriben profundidad).
// Un triángulo por capa (instancia = capa); buffer(0): x = muestras, y = modo (1 la más cercana,
// 0 la muestra 0), z = invertida.
struct VpknDepthV {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};
vertex VpknDepthV vpktaa_native_depth_vertex(uint id [[vertex_id]], uint inst [[instance_id]]) {
    const float2 p = float2((id << 1) & 2, id & 2);
    VpknDepthV o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.layer = inst;
    return o;
}
struct VpknDepthOut {
    float depth [[depth(any)]];
};
fragment VpknDepthOut vpktaa_native_depth_fragment(VpknDepthV in [[stage_in]],
                                                   depth2d_ms_array<float> src [[texture(0)]],
                                                   constant uint4& q [[buffer(0)]]) {
    const uint2 t = uint2(in.position.xy);
    float z = src.read(t, in.layer, 0);
    if (q.y == 1)
        for (uint s = 1; s < q.x; ++s) {
            const float v = src.read(t, in.layer, s);
            z = (q.z ? v > z : v < z) ? v : z;
        }
    VpknDepthOut o;
    o.depth = z;
    return o;
}
