# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# VPKEngine as shadPS4's guest CPU (AstroVisionPro): the FEX interface files of shadPS4 stay, only
# fex_guest_engine.cpp + FEXCore are replaced by aot_guest_engine.cpp + the VPKEngine runtime + the
# game's translated modules. See integrations/shadps4/README.md and astrovisionpro.patch.
#
#   VPKENGINE_DIR       this repository
#   VPKENGINE_GAME_DIR  output of tools/scripts/translate_game.(ps1|sh): *.c + vpkengine_registry.c
#                      (empty or unset: an engine with no game translated, which still builds)
#
# vpkengine_sources(<list var>) adds the sources; vpkengine_configure(<target>) the flags.

set(VPKENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "VPKEngine checkout")
set(VPKENGINE_GAME_DIR "" CACHE PATH "Translated game modules (tools/scripts/translate_game)")
# The runtime (runtime/vp_host.c) in a shared library of its own (libVPRuntime.dylib in the app's
# Frameworks) instead of in shadPS4: game packs loaded at run time bind to it (tools/scripts/
# make_game_pack). The app then links -lVPRuntime.
option(VPKENGINE_RUNTIME_SHARED "VPKEngine runtime as a separate shared library (game packs)" OFF)

function(vpkengine_sources out_var)
    set(sources "${VPKENGINE_DIR}/integrations/shadps4/aot_guest_engine.cpp")
    if (NOT VPKENGINE_RUNTIME_SHARED)
        list(APPEND sources "${VPKENGINE_DIR}/runtime/vp_host.c")
    endif()
    if (VPKENGINE_GAME_DIR)
        set(registry "${VPKENGINE_GAME_DIR}/vpkengine_registry.c")
        if (NOT EXISTS "${registry}")
            message(FATAL_ERROR "VPKENGINE_GAME_DIR=${VPKENGINE_GAME_DIR} has no vpkengine_registry.c (run tools/scripts/translate_game)")
        endif()
        # Exactly the files of the modules the registry names (NAME_files.txt when the module was
        # split, else NAME.c): leftovers of an older translation in the folder are not picked up.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${registry}")
        file(STRINGS "${registry}" decls REGEX "^extern .*VpModule vp_module_[A-Za-z0-9_]+;")
        set(game_sources "${registry}")
        foreach(decl IN LISTS decls)
            string(REGEX REPLACE "^extern .*VpModule vp_module_([A-Za-z0-9_]+);.*" "\\1" name "${decl}")
            set(list_file "${VPKENGINE_GAME_DIR}/${name}_files.txt")
            if (EXISTS "${list_file}")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${list_file}")
                file(STRINGS "${list_file}" parts)
                foreach(part IN LISTS parts)
                    string(REPLACE "\\" "/" part "${part}") # written on Windows
                    get_filename_component(part_name "${part}" NAME)
                    list(APPEND game_sources "${VPKENGINE_GAME_DIR}/${part_name}")
                endforeach()
            else()
                list(APPEND game_sources "${VPKENGINE_GAME_DIR}/${name}.c")
            endif()
        endforeach()
        list(APPEND sources ${game_sources})
        list(LENGTH decls modules)
        list(LENGTH game_sources n)
        message(STATUS "VPKEngine: ${modules} translated modules, ${n} C files from ${VPKENGINE_GAME_DIR}")
    else()
        # No game: an empty registry (written only when missing, so it is not rebuilt each time).
        set(empty "${CMAKE_CURRENT_BINARY_DIR}/vpkengine_registry_empty.c")
        if (NOT EXISTS "${empty}")
            file(WRITE "${empty}" "void vp_register_game_modules(void) {}\n")
        endif()
        list(APPEND sources "${empty}")
        message(STATUS "VPKEngine: no translated game (VPKENGINE_GAME_DIR unset)")
    endif()
    # The translated C and the runtime follow MXCSR's rounding through the host's FP environment:
    # the compiler must not fold or move FP operations across it.
    foreach(s IN LISTS sources)
        if (s MATCHES "\\.c$")
            set_source_files_properties("${s}" PROPERTIES COMPILE_OPTIONS "-O2;-frounding-math;-w"
                                        INCLUDE_DIRECTORIES "${VPKENGINE_DIR}/runtime")
        endif()
    endforeach()
    set(${out_var} ${${out_var}} ${sources} PARENT_SCOPE)
endfunction()

function(vpkengine_configure target)
    target_include_directories(${target} PRIVATE "${VPKENGINE_DIR}/runtime")
    # The rest of shadPS4 takes its FEX paths (veneers, HLE bridge, fibers, signals) unchanged.
    target_compile_definitions(${target} PRIVATE SHADPS4_ENABLE_FEX_GUEST_CPU=1 SHADPS4_GUEST_CPU_VPKENGINE=1)
endfunction()
