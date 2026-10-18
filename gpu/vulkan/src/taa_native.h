// VPKTAA nativo: el TAA como pase interno de VPVulkan (vpvk_taa_native_encode, vpvk/config.h).
//
// Por qué así (frente al VPKTAACore de Klepton, que sigue siendo la ruta por defecto):
//   * El jitter lo elige el motor (Halton(2,3) de 8 fases, una por frame resuelto) y lo aplica el
//     traductor al viewport de los draws de escena del ojo (commands.cpp taaEyePass/jitter). Mover el
//     viewport jx píxeles equivale exactamente a sumar 2·jx/W·w a la x de clip del vertex shader
//     (la transformación de viewport es afín tras la división por w; ver jitterClipOffset, con test),
//     pero no cambia el MSL: sin variantes de shader, sin subir kTranslatorVersion, sin invalidar
//     recetas ni la caché de Metal. El resultado en píxel es el mismo que el de UE/FSR 2 (offset en
//     la matriz de proyección).
//   * El motor conoce el jitter que REALMENTE aplicó a cada envío; el pase lo usa sin pasar por
//     Klepton (adiós al emparejamiento jitter/frame y a sus votos).
//   * Todo lo que sigue es matemática pura (Linux, tests): pesos de reconstrucción, Catmull-Rom,
//     reproyección de cámara y linealización de profundidad. Los kernels MSL
//     (gpu/visionos/shaders/vpktaa_native.metal) hacen exactamente estas cuentas.
#pragma once

#include <array>
#include <cstdint>

namespace vpvk::taa {

// Radical inverse de base b del índice i (i >= 1 para evitar el 0).
float halton(uint32_t i, uint32_t base);
// Jitter de la fase `frame` (8 fases, Halton(2,3) desde el índice 1), en píxeles del target,
// [-0.5, 0.5): +x derecha, +y abajo (convención de vpvk_taa_jitter).
std::array<float, 2> haltonJitter(uint64_t frame);
inline constexpr uint32_t kPhases = 8;

// Desplazamiento de la x/y de clip que equivale a mover el viewport (jx, jy) píxeles en un target
// de w x h con clip.w = cw: dx = 2·jx/W·cw, dy = -2·jy/H·cw (en Vulkan +y de framebuffer es +y de
// NDC; Metal invierte y al emitir, de ahí el signo para NDC de Metal). Solo para tests/documentación.
std::array<float, 2> jitterClipOffset(float jx, float jy, float w, float h, float cw);

// Reconstrucción: la muestra del texel k guarda el contenido de (k + 0.5 - s) en píxeles sin jitter
// (s = desplazamiento del contenido en texels del color, uniforme en el frame). Para el píxel de
// salida g el texel más cercano es g + i, i = floor(0.5 + s), y la distancia del vecino (ox, oy)
// de ese texel es (i + o - s): 9 pesos gaussianos (ajuste de Blackman-Harris, exp(-2.29·d²)),
// iguales en todo el frame, normalizados a suma 1.
struct Reconstruction {
  std::array<int32_t, 2> base{};      // i
  std::array<float, 9> weights{};     // fila mayor, (ox,oy) en -1..1
  std::array<int32_t, 2> nearest{};   // offset (-1..1) del texel más cercano al centro
};
Reconstruction reconstruction(float sx, float sy);

// Pesos Catmull-Rom (tensión 0.5) de los 4 texels alrededor de una fracción f en [0,1): suman 1.
std::array<float, 4> catmullRom(float f);
// Forma de 5 lecturas bilineales (sin las 4 esquinas) por eje: p0 = -1, p12 = w2/(w1+w2), p3 = +2
// relativos al texel base; devuelve {w0, w12, w3, offset12}.
std::array<float, 4> catmullRom5(float f);

// Profundidad Vulkan [0,1] -> distancia de vista (positiva). far 0 = infinito.
float viewDistance(float depth, float nearZ, float farZ, bool reversedZ);
// Profundidad que da el clear (nada dibujado).
inline bool emptyDepth(float d, bool reversedZ) { return reversedZ ? d <= 0.0f : d >= 1.0f; }

struct Quat {
  float x = 0, y = 0, z = 0, w = 1;
};
struct Vec3 {
  float x = 0, y = 0, z = 0;
};
// Pose de ojo (mundo <- ojo) y frustum sin jitter (tangentes l r t b positivas). Ejes OpenXR:
// -z al frente, +y arriba, +x derecha.
struct EyeView {
  Quat orientation;
  Vec3 position;
  std::array<float, 4> tangents{1, 1, 1, 1};
};
// Ojo actual -> ojo anterior: p_prev = R·p_cur + T.
struct Reprojection {
  std::array<float, 9> r{1, 0, 0, 0, 1, 0, 0, 0, 1};  // fila mayor
  Vec3 t;
};
Reprojection reprojection(const EyeView& cur, const EyeView& prev);
// Reproyecta el uv (0..1 dentro del rectángulo del ojo, fila 0 arriba) de un punto a distancia
// `dist` (0 = infinito: solo rotación) y devuelve el uv anterior y su distancia en el frame anterior.
// false si queda detrás de la cámara anterior.
bool reprojectUv(const Reprojection& m, const std::array<float, 4>& tanCur, const std::array<float, 4>& tanPrev,
                 float u, float v, float dist, float& pu, float& pv, float& prevDist);

}  // namespace vpvk::taa
