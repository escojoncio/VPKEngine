# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# VPEngine as shadPS4's guest CPU (AstroVisionPro): the FEX interface files of shadPS4 stay, only
# fex_guest_engine.cpp + FEXCore are replaced by aot_guest_engine.cpp + the VPEngine runtime + the
# game's translated modules. See integrations/shadps4/README.md and astrovisionpro.patch.
#
#   VPENGINE_DIR       this repository
#   VPENGINE_GAME_DIR  output of tools/scripts/translate_game.(ps1|sh): *.c + vpengine_registry.c
#                      (empty or unset: an engine with no game translated, which still builds)
#
# vpengine_sources(<list var>) adds the sources; vpengine_configure(<target>) the flags.

set(VPENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "VPEngine checkout")
set(VPENGINE_GAME_DIR "" CACHE PATH "Translated game modules (tools/scripts/translate_game)")

function(vpengine_sources out_var)
    set(sources
        "${VPENGINE_DIR}/integrations/shadps4/aot_guest_engine.cpp"
        "${VPENGINE_DIR}/runtime/vp_host.c"
    )
    if (VPENGINE_GAME_DIR AND EXISTS "${VPENGINE_GAME_DIR}/vpengine_registry.c")
        file(GLOB game_sources "${VPENGINE_GAME_DIR}/*.c")
        list(APPEND sources ${game_sources})
        list(LENGTH game_sources n)
        message(STATUS "VPEngine: ${n} translated C files from ${VPENGINE_GAME_DIR}")
    else()
        # No game: an empty registry.
        set(empty "${CMAKE_CURRENT_BINARY_DIR}/vpengine_registry_empty.c")
        file(WRITE "${empty}" "void vp_register_game_modules(void) {}\n")
        list(APPEND sources "${empty}")
        message(STATUS "VPEngine: no translated game (VPENGINE_GAME_DIR unset)")
    endif()
    # The translated C and the runtime follow MXCSR's rounding through the host's FP environment:
    # the compiler must not fold or move FP operations across it.
    foreach(s IN LISTS sources)
        if (s MATCHES "\\.c$")
            set_source_files_properties("${s}" PROPERTIES COMPILE_OPTIONS "-O2;-frounding-math;-w"
                                        INCLUDE_DIRECTORIES "${VPENGINE_DIR}/runtime")
        endif()
    endforeach()
    set(${out_var} ${${out_var}} ${sources} PARENT_SCOPE)
endfunction()

function(vpengine_configure target)
    target_include_directories(${target} PRIVATE "${VPENGINE_DIR}/runtime")
    # The rest of shadPS4 takes its FEX paths (veneers, HLE bridge, fibers, signals) unchanged.
    target_compile_definitions(${target} PRIVATE SHADPS4_ENABLE_FEX_GUEST_CPU=1 SHADPS4_GUEST_CPU_VPENGINE=1)
endfunction()
