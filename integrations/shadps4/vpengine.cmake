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
    if (VPENGINE_GAME_DIR)
        set(registry "${VPENGINE_GAME_DIR}/vpengine_registry.c")
        if (NOT EXISTS "${registry}")
            message(FATAL_ERROR "VPENGINE_GAME_DIR=${VPENGINE_GAME_DIR} has no vpengine_registry.c (run tools/scripts/translate_game)")
        endif()
        # Exactly the files of the modules the registry names (NAME_files.txt when the module was
        # split, else NAME.c): leftovers of an older translation in the folder are not picked up.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${registry}")
        file(STRINGS "${registry}" decls REGEX "^extern VpModule vp_module_[A-Za-z0-9_]+;")
        set(game_sources "${registry}")
        foreach(decl IN LISTS decls)
            string(REGEX REPLACE "^extern VpModule vp_module_([A-Za-z0-9_]+);.*" "\\1" name "${decl}")
            set(list_file "${VPENGINE_GAME_DIR}/${name}_files.txt")
            if (EXISTS "${list_file}")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${list_file}")
                file(STRINGS "${list_file}" parts)
                foreach(part IN LISTS parts)
                    string(REPLACE "\\" "/" part "${part}") # written on Windows
                    get_filename_component(part_name "${part}" NAME)
                    list(APPEND game_sources "${VPENGINE_GAME_DIR}/${part_name}")
                endforeach()
            else()
                list(APPEND game_sources "${VPENGINE_GAME_DIR}/${name}.c")
            endif()
        endforeach()
        list(APPEND sources ${game_sources})
        list(LENGTH decls modules)
        list(LENGTH game_sources n)
        message(STATUS "VPEngine: ${modules} translated modules, ${n} C files from ${VPENGINE_GAME_DIR}")
    else()
        # No game: an empty registry (written only when missing, so it is not rebuilt each time).
        set(empty "${CMAKE_CURRENT_BINARY_DIR}/vpengine_registry_empty.c")
        if (NOT EXISTS "${empty}")
            file(WRITE "${empty}" "void vp_register_game_modules(void) {}\n")
        endif()
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
