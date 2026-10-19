// Foveación propia (vpvk_config_set_foveation): perfil del mapa de tasas de rasterización de los pases de ojo.
//
// Todo lo de este fichero es matemática pura (Linux, tests); el MTLRasterizationRateMap, las texturas físicas y el
// desdoblado viven en gpu/metal2 (metal_backend.mm, `RenderPassDesc::rateMap`). Diseño completo en docs/VULKAN.md
// («Foveado propio»):
//   * el traductor (commands.cpp) marca con un `vpgpu::RateMap` los pases multiview de 2 vistas que identifica como
//     pase de ojo (los del registro del TAA); el resto del juego no se entera;
//   * el backend pinta ese pase en attachments físicos más pequeños con el mapa y, al terminar, los DESDOBLA a los
//     attachments lógicos del juego (compute/render propio): todo lo que viene después (post del juego, VPKTAA
//     nativo, compositor) ve texturas del tamaño lógico de siempre;
//   * perfil: zona nítida (tasa 1), periferia (tasa del borde) y caída suave entre ambas, centrado en el centro
//     óptico de cada ojo (tangentes asimétricas del frustum). Los arrays de calidad son por eje (un rate map es
//     separable): el contorno resultante es un cuadrado redondeado.
#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "vpgpu/core/commands.h"

namespace vpvk::fov {

inline constexpr uint32_t kZones = 32;     // zonas por eje (Klepton KL_VRR_ZONES usa 16; 32 da un contorno más redondo)
inline constexpr uint32_t kMaxZones = 64;  // KL_FOVEATION_MAX_ZONES
inline constexpr uint16_t kViews = 2;      // solo pases estéreo de 2 vistas (capa del mapa = vista = viewport index)

// Perfil resuelto de un ajuste (vpvk_foveation).
struct Shape {
  float sharp = 0.35f;      // radio de la zona nítida, fracción de la semi-extensión (0.05..1)
  float periphery = 0.5f;   // tasa por eje en el borde (0.1..1; 1 = sin reducción)
  float height = 0.0f;      // desplazamiento vertical del centro, fracción del alto (+ arriba)
  float fade = 0.15f;       // caída 1 -> periferia sobre este tramo exterior del radio nítido (PTVisionPro: 0.15)
  bool off = true;          // modo apagado
};
// `mode` es vpvk_foveation_mode (0 apagado, 1 M2, 2 M5, 3 personalizado). Los presets M2/M5 ignoran los tres valores
// (salvo `height`, que sigue valiendo); el personalizado usa los tres.
Shape shapeOf(int mode, float sharp, float periphery, float height);
// Presets = los de PTVisionPro (PTSettings.defaults): M2 zona nítida 60 grados, periferia 10 %, centro 10 grados bajo el
// eje; M5 70 grados, 15 %, 10 grados bajo el eje. Grados a fracción de la semi-extensión como su VrrCenterFraction
// (tan(grados/2)/1.2); desnivel como fracción del alto: tan(10 grados)/(2*1.2).
Shape presetM2();
Shape presetM5();
const char* modeName(int mode);

// Tangentes del frustum de cada ojo (l r t b, positivas) si el anfitrión las dio.
struct Eyes {
  bool known = false;
  std::array<std::array<float, 4>, 2> tangents{{{1, 1, 1, 1}, {1, 1, 1, 1}}};
};
// Centro óptico del ojo en fracciones de su extensión desde la izquierda y desde arriba: la dirección de la mirada
// al frente cae en x = l/(l+r), y = t/(t+b) (sin tangentes: el centro, 0.5).
struct Center {
  float x = 0.5f, y = 0.5f;
};
Center opticalCenter(const std::array<float, 4>& tangents);

// Calidad (0..1) de `zones` zonas iguales de un eje cuyo centro está en la fracción `center` (0..1): 1 dentro de la
// zona nítida, `periphery` en el borde (a cada lado, su propio borde), caída suave (smootherstep) entre medias.
// `fade` < 0 (por defecto): caída suave desde `sharp` hasta el borde de la imagen. `fade` >= 0 (perfil PTVisionPro): 1
// hasta sharp*(1-fade), caída smootherstep hasta `periphery` en `sharp` y periferia constante más allá.
void axisQuality(float* q, uint32_t zones, float center, float sharp, float periphery, float fade = -1.0f);

// Fracción de fragmentos que quedan (media del producto de calidades por eje, media de las capas).
float keptFraction(const vpgpu::RateMap& m);

// Mapa de un ojo lógico de w x h con `layers` vistas (kViews). Sin `eyes.known`: centro geométrico en las dos.
std::shared_ptr<vpgpu::RateMap> buildRateMap(uint32_t w, uint32_t h, uint16_t layers, const Shape& shape, const Eyes& eyes,
                                             uint32_t zones = kZones);

}  // namespace vpvk::fov
