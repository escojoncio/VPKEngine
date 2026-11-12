#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* params;
};

struct VpIn {
    float4 a [[user(loc0)]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]]) {
    VpOut vp_out;
    const uint v2 = uint(vp_in.a.x);
    while (true) {
        const float v6 = vp_in.a.x / (*(constant float*)(vp_args.params + v2));
        const bool v8 = v6 < 0.0f;
        if (v8) {
            vp_out.color = vp_in.a;
        } else {
            const uint v10 = v2 + 16u;
            const float v12 = vp_in.a.x / (*(constant float*)(vp_args.params + v10));
            const bool v14 = v12 < 1.0f;
            if (v14) {
                vp_out.color = vp_in.a;
            } else {
                const uint v16 = v2 + 32u;
                const float v18 = vp_in.a.x / (*(constant float*)(vp_args.params + v16));
                const bool v20 = v18 < 2.0f;
                if (v20) {
                    vp_out.color = vp_in.a;
                } else {
                    const uint v22 = v2 + 48u;
                    const float v24 = vp_in.a.x / (*(constant float*)(vp_args.params + v22));
                    const bool v26 = v24 < 3.0f;
                    if (v26) {
                        vp_out.color = vp_in.a;
                    } else {
                        const uint v28 = v2 + 64u;
                        const float v30 = vp_in.a.x / (*(constant float*)(vp_args.params + v28));
                        const bool v32 = v30 < 4.0f;
                        if (v32) {
                            vp_out.color = vp_in.a;
                        } else {
                            const uint v34 = v2 + 80u;
                            const float v36 = vp_in.a.x / (*(constant float*)(vp_args.params + v34));
                            const bool v38 = v36 < 5.0f;
                            if (v38) {
                                vp_out.color = vp_in.a;
                                break;
                            } else {
                                const uint v40 = v2 + 96u;
                                const float v42 = vp_in.a.x / (*(constant float*)(vp_args.params + v40));
                                const bool v44 = v42 < 6.0f;
                                do {
                                    if (v44) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v46 = v2 + 112u;
                                    const float v48 = vp_in.a.x / (*(constant float*)(vp_args.params + v46));
                                    const bool v50 = v48 < 7.0f;
                                    if (v50) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v52 = v2 + 128u;
                                    const float v54 = vp_in.a.x / (*(constant float*)(vp_args.params + v52));
                                    const bool v56 = v54 < 8.0f;
                                    if (v56) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v58 = v2 + 144u;
                                    const float v60 = vp_in.a.x / (*(constant float*)(vp_args.params + v58));
                                    const bool v62 = v60 < 9.0f;
                                    if (v62) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v64 = v2 + 160u;
                                    const float v66 = vp_in.a.x / (*(constant float*)(vp_args.params + v64));
                                    const bool v68 = v66 < 10.0f;
                                    if (v68) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v70 = v2 + 176u;
                                    const float v72 = vp_in.a.x / (*(constant float*)(vp_args.params + v70));
                                    const bool v74 = v72 < 11.0f;
                                    if (v74) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v76 = v2 + 192u;
                                    const float v78 = vp_in.a.x / (*(constant float*)(vp_args.params + v76));
                                    const bool v80 = v78 < 12.0f;
                                    if (v80) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v82 = v2 + 208u;
                                    const float v84 = vp_in.a.x / (*(constant float*)(vp_args.params + v82));
                                    const bool v86 = v84 < 13.0f;
                                    if (v86) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v88 = v2 + 224u;
                                    const float v90 = vp_in.a.x / (*(constant float*)(vp_args.params + v88));
                                    const bool v92 = v90 < 14.0f;
                                    if (v92) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v94 = v2 + 240u;
                                    const float v96 = vp_in.a.x / (*(constant float*)(vp_args.params + v94));
                                    const bool v98 = v96 < 15.0f;
                                    if (v98) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v100 = v2 + 256u;
                                    const float v102 = vp_in.a.x / (*(constant float*)(vp_args.params + v100));
                                    const bool v104 = v102 < 16.0f;
                                    if (v104) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v106 = v2 + 272u;
                                    const float v108 = vp_in.a.x / (*(constant float*)(vp_args.params + v106));
                                    const bool v110 = v108 < 17.0f;
                                    if (v110) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v112 = v2 + 288u;
                                    const float v114 = vp_in.a.x / (*(constant float*)(vp_args.params + v112));
                                    const bool v116 = v114 < 18.0f;
                                    if (v116) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v118 = v2 + 304u;
                                    const float v120 = vp_in.a.x / (*(constant float*)(vp_args.params + v118));
                                    const bool v122 = v120 < 19.0f;
                                    if (v122) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v124 = v2 + 320u;
                                    const float v126 = vp_in.a.x / (*(constant float*)(vp_args.params + v124));
                                    const bool v128 = v126 < 20.0f;
                                    if (v128) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v130 = v2 + 336u;
                                    const float v132 = vp_in.a.x / (*(constant float*)(vp_args.params + v130));
                                    const bool v134 = v132 < 21.0f;
                                    if (v134) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v136 = v2 + 352u;
                                    const float v138 = vp_in.a.x / (*(constant float*)(vp_args.params + v136));
                                    const bool v140 = v138 < 22.0f;
                                    if (v140) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v142 = v2 + 368u;
                                    const float v144 = vp_in.a.x / (*(constant float*)(vp_args.params + v142));
                                    const bool v146 = v144 < 23.0f;
                                    if (v146) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v148 = v2 + 384u;
                                    const float v150 = vp_in.a.x / (*(constant float*)(vp_args.params + v148));
                                    const bool v152 = v150 < 24.0f;
                                    if (v152) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v154 = v2 + 400u;
                                    const float v156 = vp_in.a.x / (*(constant float*)(vp_args.params + v154));
                                    const bool v158 = v156 < 25.0f;
                                    if (v158) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v160 = v2 + 416u;
                                    const float v162 = vp_in.a.x / (*(constant float*)(vp_args.params + v160));
                                    const bool v164 = v162 < 26.0f;
                                    if (v164) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v166 = v2 + 432u;
                                    const float v168 = vp_in.a.x / (*(constant float*)(vp_args.params + v166));
                                    const bool v170 = v168 < 27.0f;
                                    if (v170) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v172 = v2 + 448u;
                                    const float v174 = vp_in.a.x / (*(constant float*)(vp_args.params + v172));
                                    const bool v176 = v174 < 28.0f;
                                    if (v176) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v178 = v2 + 464u;
                                    const float v180 = vp_in.a.x / (*(constant float*)(vp_args.params + v178));
                                    const bool v182 = v180 < 29.0f;
                                    if (v182) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v184 = v2 + 480u;
                                    const float v186 = vp_in.a.x / (*(constant float*)(vp_args.params + v184));
                                    const bool v188 = v186 < 30.0f;
                                    if (v188) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v190 = v2 + 496u;
                                    const float v192 = vp_in.a.x / (*(constant float*)(vp_args.params + v190));
                                    const bool v194 = v192 < 31.0f;
                                    if (v194) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v196 = v2 + 512u;
                                    const float v198 = vp_in.a.x / (*(constant float*)(vp_args.params + v196));
                                    const bool v200 = v198 < 32.0f;
                                    if (v200) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v202 = v2 + 528u;
                                    const float v204 = vp_in.a.x / (*(constant float*)(vp_args.params + v202));
                                    const bool v206 = v204 < 33.0f;
                                    if (v206) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v208 = v2 + 544u;
                                    const float v210 = vp_in.a.x / (*(constant float*)(vp_args.params + v208));
                                    const bool v212 = v210 < 34.0f;
                                    if (v212) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v214 = v2 + 560u;
                                    const float v216 = vp_in.a.x / (*(constant float*)(vp_args.params + v214));
                                    const bool v218 = v216 < 35.0f;
                                    if (v218) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v220 = v2 + 576u;
                                    const float v222 = vp_in.a.x / (*(constant float*)(vp_args.params + v220));
                                    const bool v224 = v222 < 36.0f;
                                    if (v224) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v226 = v2 + 592u;
                                    const float v228 = vp_in.a.x / (*(constant float*)(vp_args.params + v226));
                                    const bool v230 = v228 < 37.0f;
                                    if (v230) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v232 = v2 + 608u;
                                    const float v234 = vp_in.a.x / (*(constant float*)(vp_args.params + v232));
                                    const bool v236 = v234 < 38.0f;
                                    if (v236) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v238 = v2 + 624u;
                                    const float v240 = vp_in.a.x / (*(constant float*)(vp_args.params + v238));
                                    const bool v242 = v240 < 39.0f;
                                    if (v242) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v244 = v2 + 640u;
                                    const float v246 = vp_in.a.x / (*(constant float*)(vp_args.params + v244));
                                    const bool v248 = v246 < 40.0f;
                                    if (v248) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v250 = v2 + 656u;
                                    const float v252 = vp_in.a.x / (*(constant float*)(vp_args.params + v250));
                                    const bool v254 = v252 < 41.0f;
                                    if (v254) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v256 = v2 + 672u;
                                    const float v258 = vp_in.a.x / (*(constant float*)(vp_args.params + v256));
                                    const bool v260 = v258 < 42.0f;
                                    if (v260) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v262 = v2 + 688u;
                                    const float v264 = vp_in.a.x / (*(constant float*)(vp_args.params + v262));
                                    const bool v266 = v264 < 43.0f;
                                    if (v266) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v268 = v2 + 704u;
                                    const float v270 = vp_in.a.x / (*(constant float*)(vp_args.params + v268));
                                    const bool v272 = v270 < 44.0f;
                                    if (v272) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v274 = v2 + 720u;
                                    const float v276 = vp_in.a.x / (*(constant float*)(vp_args.params + v274));
                                    const bool v278 = v276 < 45.0f;
                                    if (v278) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v280 = v2 + 736u;
                                    const float v282 = vp_in.a.x / (*(constant float*)(vp_args.params + v280));
                                    const bool v284 = v282 < 46.0f;
                                    if (v284) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v286 = v2 + 752u;
                                    const float v288 = vp_in.a.x / (*(constant float*)(vp_args.params + v286));
                                    const bool v290 = v288 < 47.0f;
                                    if (v290) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v292 = v2 + 768u;
                                    const float v294 = vp_in.a.x / (*(constant float*)(vp_args.params + v292));
                                    const bool v296 = v294 < 48.0f;
                                    if (v296) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v298 = v2 + 784u;
                                    const float v300 = vp_in.a.x / (*(constant float*)(vp_args.params + v298));
                                    const bool v302 = v300 < 49.0f;
                                    if (v302) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v304 = v2 + 800u;
                                    const float v306 = vp_in.a.x / (*(constant float*)(vp_args.params + v304));
                                    const bool v308 = v306 < 50.0f;
                                    if (v308) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v310 = v2 + 816u;
                                    const float v312 = vp_in.a.x / (*(constant float*)(vp_args.params + v310));
                                    const bool v314 = v312 < 51.0f;
                                    if (v314) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v316 = v2 + 832u;
                                    const float v318 = vp_in.a.x / (*(constant float*)(vp_args.params + v316));
                                    const bool v320 = v318 < 52.0f;
                                    if (v320) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v322 = v2 + 848u;
                                    const float v324 = vp_in.a.x / (*(constant float*)(vp_args.params + v322));
                                    const bool v326 = v324 < 53.0f;
                                    if (v326) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v328 = v2 + 864u;
                                    const float v330 = vp_in.a.x / (*(constant float*)(vp_args.params + v328));
                                    const bool v332 = v330 < 54.0f;
                                    if (v332) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v334 = v2 + 880u;
                                    const float v336 = vp_in.a.x / (*(constant float*)(vp_args.params + v334));
                                    const bool v338 = v336 < 55.0f;
                                    if (v338) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v340 = v2 + 896u;
                                    const float v342 = vp_in.a.x / (*(constant float*)(vp_args.params + v340));
                                    const bool v344 = v342 < 56.0f;
                                    if (v344) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v346 = v2 + 912u;
                                    const float v348 = vp_in.a.x / (*(constant float*)(vp_args.params + v346));
                                    const bool v350 = v348 < 57.0f;
                                    if (v350) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v352 = v2 + 928u;
                                    const float v354 = vp_in.a.x / (*(constant float*)(vp_args.params + v352));
                                    const bool v356 = v354 < 58.0f;
                                    if (v356) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v358 = v2 + 944u;
                                    const float v360 = vp_in.a.x / (*(constant float*)(vp_args.params + v358));
                                    const bool v362 = v360 < 59.0f;
                                    if (v362) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v364 = v2 + 960u;
                                    const float v366 = vp_in.a.x / (*(constant float*)(vp_args.params + v364));
                                    const bool v368 = v366 < 60.0f;
                                    if (v368) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v370 = v2 + 976u;
                                    const float v372 = vp_in.a.x / (*(constant float*)(vp_args.params + v370));
                                    const bool v374 = v372 < 61.0f;
                                    if (v374) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v376 = v2 + 992u;
                                    const float v378 = vp_in.a.x / (*(constant float*)(vp_args.params + v376));
                                    const bool v380 = v378 < 62.0f;
                                    if (v380) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v382 = v2 + 1008u;
                                    const float v384 = vp_in.a.x / (*(constant float*)(vp_args.params + v382));
                                    const bool v386 = v384 < 63.0f;
                                    if (v386) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v388 = v2 + 1024u;
                                    const float v390 = vp_in.a.x / (*(constant float*)(vp_args.params + v388));
                                    const bool v392 = v390 < 64.0f;
                                    if (v392) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v394 = v2 + 1040u;
                                    const float v396 = vp_in.a.x / (*(constant float*)(vp_args.params + v394));
                                    const bool v398 = v396 < 65.0f;
                                    if (v398) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v400 = v2 + 1056u;
                                    const float v402 = vp_in.a.x / (*(constant float*)(vp_args.params + v400));
                                    const bool v404 = v402 < 66.0f;
                                    if (v404) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v406 = v2 + 1072u;
                                    const float v408 = vp_in.a.x / (*(constant float*)(vp_args.params + v406));
                                    const bool v410 = v408 < 67.0f;
                                    if (v410) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v412 = v2 + 1088u;
                                    const float v414 = vp_in.a.x / (*(constant float*)(vp_args.params + v412));
                                    const bool v416 = v414 < 68.0f;
                                    if (v416) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    const uint v418 = v2 + 1104u;
                                    const float v420 = vp_in.a.x / (*(constant float*)(vp_args.params + v418));
                                    const bool v422 = v420 < 69.0f;
                                    if (v422) {
                                        vp_out.color = vp_in.a;
                                        break;
                                    }
                                    discard_fragment();
                                } while (false);
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    return vp_out;
}
