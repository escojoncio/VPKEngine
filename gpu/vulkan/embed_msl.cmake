# Escribe OUT = cabecera C++ con el contenido de IN como cadena cruda (kVpktaaNativeMsl).
file(READ ${IN} body)
file(WRITE ${OUT} "// Generado desde ${IN} (embed_msl.cmake). No editar.\n#pragma once\nstatic const char kVpktaaNativeMsl[] = R\"VPKN(${body})VPKN\";\n")
