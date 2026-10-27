#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    const device uchar* esgs_ring0;
    device uchar* gsvs_ring1;
    device uchar* gs_index2;
    constant uchar* geom_params0;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_global_id [[thread_position_in_grid]]) {
    bool l0{};
    bool l1{};
    bool l2{};
    uint l3{};
    uint l4{};
    uint l5{};
    uint l6{};
    uint l7{};
    uint l8{};
    uint l9{};
    uint l10{};
    uint l11{};
    uint l12{};
    uint l13{};
    uint l14{};
    uint l15{};
    uint l16{};
    uint l17{};
    uint l18{};
    bool l19{};
    uint l20{};
    uint l21{};
    uint l22{};
    bool l23{};
    uint l24{};
    uint l25{};
    uint l26{};
    uint l27{};
    uint l28{};
    uint l29{};
    uint l30{};
    uint l31{};
    uint l32{};
    uint l33{};
    uint l34{};
    uint l35{};
    uint l36{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    const uint v7 = in_global_id.x & 63u;
    const bool v12 = in_global_id.x >= (*(constant uint*)(vp_args.geom_params0 + 20u));
    if (v12) {
        return;
    }
    const uint v15 = *(constant uint*)(vp_args.geom_params0 + 16u);
    const uint v16 = in_global_id.x / v15;
    const uint v17 = v16 * v15;
    const uint v18 = in_global_id.x - v17;
    const uint v23 = v18 * 3u;
    const uint v24 = v16 * (*(constant uint*)(vp_args.geom_params0 + 4u));
    const uint v25 = v24 + v23;
    const uint v27 = v25 & 63u;
    const uint v30 = v25 >> 6u;
    const uint v31 = v30 * 512u;
    const uint v32 = v31 + v27;
    l4 = v32;
    const uint v37 = v23 + 1u;
    const uint v39 = v24 + v37;
    const uint v41 = v39 & 63u;
    const uint v44 = v39 >> 6u;
    const uint v45 = v44 * 512u;
    const uint v46 = v45 + v41;
    l5 = v46;
    const uint v51 = v23 + 2u;
    const uint v53 = v24 + v51;
    const uint v55 = v53 & 63u;
    const uint v58 = v53 >> 6u;
    const uint v59 = v58 * 512u;
    const uint v60 = v59 + v55;
    l6 = v60;
    l7 = 0u;
    l8 = 0u;
    l9 = 0u;
    l10 = v18;
    l11 = 0u;
    const uint v65 = in_global_id.x >> 6u;
    const uint v67 = v65 * 2048u;
    l12 = v67;
    l13 = 0u;
    const uint v70 = v65 * 512u;
    l14 = v70;
    l15 = in_global_id.x;
    l16 = 0u;
    l17 = 0u;
    l18 = 0u;
    l19 = false;
    uint vp_guard0 = 0u;
    while (true) {
        if (++vp_guard0 > 8192u) break;
        const uint v77 = l10;
        const uint v79 = 1u & v77;
        const uint v80 = l20;
        const bool v81 = l0;
        const uint v82 = v81 ? v79 : v80;
        l20 = v82;
        const uint v83 = l20;
        const bool v85 = 0u == v83;
        const bool v86 = l0;
        const bool v87 = v85 && v86;
        l1 = v87;
        const uint v90 = v87 ? 1u : 0u;
        l21 = v90;
        l22 = 0u;
        const bool v92 = l0;
        const bool v93 = l1;
        l23 = v92;
        const uint v96 = v92 ? 1u : 0u;
        l24 = v96;
        l25 = 0u;
        const bool v98 = v93 && v92;
        l0 = v98;
        l2 = v98;
        const uint v102 = l4;
        const uint v103 = v102 << 2u;
        const uint v104 = l26;
        const bool v105 = l0;
        const uint v106 = v105 ? v103 : v104;
        l26 = v106;
        l27 = 0u;
        const uint v108 = l27;
        const uint v110 = l26;
        const uint v112 = v110 + v108;
        const uint v116 = *(const device uint*)(vp_args.esgs_ring0 + v112);
        const uint v117 = l28;
        const bool v118 = l0;
        const uint v119 = v118 ? v116 : v117;
        l28 = v119;
        l27 = 256u;
        const uint v121 = l27;
        const uint v123 = l26;
        const uint v125 = v123 + v121;
        const uint v129 = *(const device uint*)(vp_args.esgs_ring0 + v125);
        const uint v130 = l29;
        const bool v131 = l0;
        const uint v132 = v131 ? v129 : v130;
        l29 = v132;
        l27 = 512u;
        const uint v134 = l27;
        const uint v136 = l26;
        const uint v138 = v136 + v134;
        const uint v142 = *(const device uint*)(vp_args.esgs_ring0 + v138);
        const uint v143 = l30;
        const bool v144 = l0;
        const uint v145 = v144 ? v142 : v143;
        l30 = v145;
        l27 = 768u;
        const uint v147 = l27;
        const uint v149 = l26;
        const uint v151 = v149 + v147;
        const uint v155 = *(const device uint*)(vp_args.esgs_ring0 + v151);
        const uint v156 = l31;
        const bool v157 = l0;
        const uint v158 = v157 ? v155 : v156;
        l31 = v158;
        l27 = 1024u;
        const uint v160 = l27;
        const uint v162 = l26;
        const uint v164 = v162 + v160;
        const uint v168 = *(const device uint*)(vp_args.esgs_ring0 + v164);
        const uint v169 = l32;
        const bool v170 = l0;
        const uint v171 = v170 ? v168 : v169;
        l32 = v171;
        l27 = 1280u;
        const uint v173 = l27;
        const uint v175 = l26;
        const uint v177 = v175 + v173;
        const uint v181 = *(const device uint*)(vp_args.esgs_ring0 + v177);
        const uint v182 = l33;
        const bool v183 = l0;
        const uint v184 = v183 ? v181 : v182;
        l33 = v184;
        l27 = 1536u;
        const uint v186 = l27;
        const uint v188 = l26;
        const uint v190 = v188 + v186;
        const uint v194 = *(const device uint*)(vp_args.esgs_ring0 + v190);
        const uint v195 = l34;
        const bool v196 = l0;
        const uint v197 = v196 ? v194 : v195;
        l34 = v197;
        l27 = 1792u;
        const uint v199 = l27;
        const uint v201 = l26;
        const uint v203 = v201 + v199;
        const uint v207 = *(const device uint*)(vp_args.esgs_ring0 + v203);
        const uint v208 = l35;
        const bool v209 = l0;
        const uint v210 = v209 ? v207 : v208;
        l35 = v210;
        const uint v211 = l12;
        const uint v220 = v7 >> 4u;
        const uint v221 = v220 * 512u;
        const uint v224 = v7 & 15u;
        const uint v225 = v224 * 4u;
        const uint v226 = v225 + v221;
        const uint v227 = v211 + v226;
        const bool v228 = l0;
        if (v228) {
            const uint v234 = l28;
            *(device uint*)(vp_args.gsvs_ring1 + v227) = v234;
        }
        const uint v235 = l12;
        const uint v251 = v235 + v226;
        const bool v252 = l0;
        if (v252) {
            const uint v257 = v251 + 64u;
            const uint v258 = l29;
            *(device uint*)(vp_args.gsvs_ring1 + v257) = v258;
        }
        const uint v259 = l12;
        const uint v275 = v259 + v226;
        const bool v276 = l0;
        if (v276) {
            const uint v281 = v275 + 128u;
            const uint v282 = l30;
            *(device uint*)(vp_args.gsvs_ring1 + v281) = v282;
        }
        const uint v283 = l12;
        const uint v299 = v283 + v226;
        const bool v300 = l0;
        if (v300) {
            const uint v305 = v299 + 192u;
            const uint v306 = l31;
            *(device uint*)(vp_args.gsvs_ring1 + v305) = v306;
        }
        const uint v307 = l12;
        const uint v323 = v307 + v226;
        const bool v324 = l0;
        if (v324) {
            const uint v329 = v323 + 256u;
            const uint v330 = l32;
            *(device uint*)(vp_args.gsvs_ring1 + v329) = v330;
        }
        const uint v331 = l12;
        const uint v347 = v331 + v226;
        const bool v348 = l0;
        if (v348) {
            const uint v353 = v347 + 320u;
            const uint v354 = l33;
            *(device uint*)(vp_args.gsvs_ring1 + v353) = v354;
        }
        const uint v355 = l12;
        const uint v371 = v355 + v226;
        const bool v372 = l0;
        if (v372) {
            const uint v377 = v371 + 384u;
            const uint v378 = l34;
            *(device uint*)(vp_args.gsvs_ring1 + v377) = v378;
        }
        const uint v379 = l12;
        const uint v395 = v379 + v226;
        const bool v396 = l0;
        if (v396) {
            const uint v401 = v395 + 448u;
            const uint v402 = l35;
            *(device uint*)(vp_args.gsvs_ring1 + v401) = v402;
        }
        const bool v403 = l0;
        if (v403) {
            const uint v404 = l16;
            const bool v406 = v404 < 1u;
            if (v406) {
                const uint v410 = v220 * 128u;
                const uint v413 = v224 + v410;
                const uint v415 = v404 * 16u;
                const uint v416 = l14;
                const uint v417 = v416 + v415;
                const uint v418 = v417 + v413;
                const uint v420 = v404 + 1u;
                l16 = v420;
                const uint v421 = l15;
                const uint v422 = l18;
                const uint v423 = v421 + v422;
                const uint v427 = v423 * 4u;
                *(device uint*)(vp_args.gs_index2 + v427) = v418;
                const uint v429 = v422 + 1u;
                l18 = v429;
            }
        }
        const bool v430 = l0;
        const bool v431 = l23;
        const bool v432 = v430 || v431;
        l0 = v432;
        l2 = v432;
        l19 = true;
        break;
        break;
    }
    const uint v435 = l18;
    l36 = v435;
    const uint v437 = l15;
    uint vp_guard1 = 0u;
    while (true) {
        if (++vp_guard1 > 8192u) break;
        const uint v438 = l36;
        const bool v440 = v438 >= 1u;
        if (v440) {
            break;
        }
        const uint v443 = v437 + v438;
        const uint v444 = v443 * 4u;
        *(device uint*)(vp_args.gs_index2 + v444) = 4294967295u;
        const uint v446 = v438 + 1u;
        l36 = v446;
    }
}
