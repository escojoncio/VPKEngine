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


// ---- Máscara de UI («reactive mask» de FSR 2 / «responsive AA» de Unreal) --------------------------
// El TAA solo debe actuar sobre el 3D: la UI/2D dibujada dentro del buffer del ojo (menús, textos,
// HUD en world-space) se dibuja sin jitter y marca cobertura en una máscara R8 del tamaño del ojo;
// el pase TAA usa casi solo el frame actual donde la máscara vale > 0 (sin historia, sin
// reconstrucción, sin clamp ni RCAS).
//
// Heurística (por draw, en orden de dibujo dentro de la escena; ver DrawClassifier):
//   * pantalla completa (triángulo/quad generado por vertex id)      -> Other  (ni jitter ni máscara)
//   * escribe profundidad                                            -> Scene  (geometría opaca; jitter)
//   * no escribe profundidad, YA se vio geometría opaca en la escena y
//     (mezcla alfa en algún attachment O no hay test de profundidad)  -> Ui     (sin jitter, marca máscara)
//   * el resto con test de profundidad (cielo, transparentes antes de lo opaco) -> Scene
//   * el resto sin test (fondos antes de lo opaco)                   -> Other
// Máscara: attachment extra R8Unorm en el índice kUiMaskSlot (7) SOLO de los pases de ojo que tienen
// algún draw de UI (pre-escaneo en commands.cpp uiPass), clear a 0 y store; el fragment shader de cada
// pipeline de UI se retraduce con una salida extra (SpirvLowerOptions::uiMaskSlot: alfa del color 0
// ×2 saturado si el draw mezcla por alfa, 1 si no) con blend Max (la cobertura acumula el máximo);
// los draws 3D del mismo pase usan una variante del pipeline con ese attachment y write mask 0.
// Coste (Quest-like 2208x2064 por ojo, 2 ojos, 90 Hz): textura 9 MB; ~9 MB de escritura por pase con
// máscara (~0.8 GB/s) + ~9 MB de lectura en el pase TAA (~0.8 GB/s); +1 B/px de memoria de tile; un
// color attachment menos fusionable con el pase vecino (store/load extra solo en frames con UI);
// cero en frames o pases sin UI; draws de UI: una salida más (despreciable). Compilación: +1 PSO por
// pipeline usado en un pase con UI (+1 traducción MSL por pipeline de UI), recogidos en las recetas
// (prewarm del siguiente arranque). Alternativas descartadas: reutilizar el alfa del buffer (el
// compositor y los juegos lo usan), stencil (no hay en D16/D32, choca con el stencil del juego, y
// no admite cobertura parcial), un pase extra de máscara (obliga a re-rasterizar o a partir el pase).
// Falsos positivos conocidos: transparentes 3D (partículas, cristal) con mezcla y test sin escritura
// se tratan como UI (sin historia: sin antialiasing temporal, no se corrompen); un menú dibujado
// ANTES de lo opaco no se detecta.
struct DrawFacts {
  bool depthTest = false, depthWrite = false, blended = false, fullscreen = false;
};
enum class DrawClass : uint8_t { Scene, Ui, Other };
class DrawClassifier {
 public:
  // Un pase de escena nuevo (la profundidad se limpia) empieza sin geometría opaca; uno que
  // carga la profundidad (overlay de UI sobre la escena) la hereda.
  void beginPass(bool depthLoaded) {
    if (!depthLoaded) sawOpaque_ = false;
  }
  DrawClass classify(const DrawFacts& f) {
    if (f.fullscreen) return DrawClass::Other;
    if (f.depthWrite) {
      sawOpaque_ = true;
      return DrawClass::Scene;
    }
    const bool candidate = f.blended || !f.depthTest;
    if (!candidate) return DrawClass::Scene;  // opaco con test pero sin escritura (cielo, decales)
    if (sawOpaque_) return DrawClass::Ui;
    return f.depthTest ? DrawClass::Scene : DrawClass::Other;
  }
  bool sawOpaque() const { return sawOpaque_; }

 private:
  bool sawOpaque_ = false;
};

// Attachment de color de la máscara en el pase de ojo (R8Unorm). El último índice: ninguna salida de
// un fragment shader real llega ahí, y un pase que ya lo usa se queda sin máscara.
inline constexpr int kUiMaskSlot = 7;

// Contadores por frame (log «vk.taa uimask»).
struct UiMaskCounters {
  uint32_t scene = 0, ui = 0, other = 0, passes = 0, maskedPasses = 0;
};

// Valor que escribe un draw de UI: con mezcla alfa, el alfa del color 0 (×kUiMaskAlphaGain, tope 1:
// el borde de un glifo con alfa 0.5 ya cuenta entero); sin él, 1. Réplica del código inyectado en el
// fragment shader (gpu/shader/spirv lower.cpp, SpirvLowerOptions::uiMaskSlot).
inline constexpr float kUiMaskAlphaGain = 2.0f;
float uiMaskFromAlpha(float alpha);
// Mezcla del pase TAA con máscara m (0 = 3D, 1 = UI): la base de la reconstrucción se acerca al
// texel actual SIN jitter y el peso de la historia cae a keep·(1-m). Réplica de vpktaa_native_resolve.
float maskedBase(float recon, float current, float m);
float maskedHistoryWeight(float keep, float m);
// m <= kUiMaskEps se trata como 0 (ruido de cuantización de R8).
inline constexpr float kUiMaskEps = 1.0f / 255.0f;

// ---- Profundidad del pase de ojo cuando el host no la da (vpgles, OpenXR) ------------------------
// El Translator anota de cada pase de ojo su color, su profundidad, el valor con que se limpió y cuántos
// draws probaron la profundidad con GREATER (profundidad invertida) o LESS (estándar). `reversedZ` decide
// la convención: manda el recuento de comparaciones; si empata o no hay, el valor de limpieza (< 0.5 =
// invertida); si tampoco, `fallback`.
bool reversedZ(bool cleared, float clearValue, uint32_t greater, uint32_t less, bool fallback);

// ---- RCAS (Robust Contrast Adaptive Sharpening, el afilado de FSR 1/2): kernel vpktaa_native_rcas ------
// El afilado del pipeline: un pase propio, siempre activo en todos los modos de AA (VPKTAA, MSAA del
// juego, Off) y rutas, con intensidad 0..100 % por juego (0 = el pase se omite, sin coste). Implementación
// propia del algoritmo publicado (sin código de AMD): cruz de 5 muestras (b arriba, d izq., e centro,
// f der., h abajo), por canal, en espacio perceptual (gamma 2.0 = sqrt del lineal; el kernel linealiza el
// ojo sRGB al leer y vuelve a sRGB al escribir):
//   mn = min(b,d,f,h), mx = max(b,d,f,h)
//   hitMin = min(mn, e) / (4·mx)             lóbulo negativo mayor sin quemar por debajo de 0
//   hitMax = (1 - max(mx, e)) / (4·mn - 4)   y sin pasarse de 1: el resultado no rebasa el entorno (sin ringing)
//   lobe = max(-hitMin, hitMax) (máx. de los 3 canales), acotado a [-0.1875, 0], × exp2(-stops)
//   out = sat((lobe·(b+d+f+h) + e) / (4·lobe + 1))
// Nitidez en «stops» como en FSR: stops = 2·(1 - intensidad), peso exp2(-stops): 100 % = 0 stops (lóbulo
// completo), 50 % = 1 stop (la mitad), 0 % = apagado (no es 2 stops: el pase no corre). Denoise opcional
// (VPVK_RCAS_DENOISE=1, para imagen con grano): el lóbulo se atenúa hasta la mitad donde el centro se
// separa de la media de la cruz (ruido) en proporción al rango local. Sin escalado: el ojo sale al tamaño
// de render (el desplegado del foveado lo hace el hardware con su rate map; la resolución de render la
// escala el compositor). La UI (máscara R8) queda intacta: out = mix(rcas, e, m).
inline constexpr float kRcasLimit = 0.25f - 1.0f / 16.0f;  // 0.1875
// Intensidad de la app (0..1; NaN = 0). 0 apaga el pase.
inline float rcasClamp(float s) { return s != s ? 0.0f : s < 0.0f ? 0.0f : s > 1.0f ? 1.0f : s; }
// Peso exp2(-stops) de la intensidad (1 con 100 %, 0.5 con 50 %).
float rcasConstant(float intensity);
// Un píxel de la cruz: c[0..4] = {b, d, e, f, h}, cada uno rgb perceptual 0..1. Con intensidad 0: e exacto.
std::array<float, 3> rcasPixel(const std::array<std::array<float, 3>, 5>& c, float intensity, bool denoise);

}  // namespace vpvk::taa
