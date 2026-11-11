# VPWin: aplicaciones Windows (Steam) con Wine 11 + CPU/GPU de VPKEngine

Objetivo: instalar el cliente real de Steam, instalar juegos y jugarlos, incluidos los que arrancan
desde un launcher que lanza el juego. Estado: aparcado el 2026-11-11 (coste: meses hasta un primer
juego); solo diseño + medición, nada del backend escrito.

## Arquitectura elegida (pendiente de confirmar: mete Wine, LGPL-2.1+, como biblioteca aparte)
- Capa API = Wine 11.0 sin tocar por encima de la costura. Sus DLL PE **x86-64** y las de la app se
  traducen con `vpaot` (sin JIT); el lado Unix de Wine (ntdll.so, win32u.so, wineserver) va nativo.
- Corte en las dos puertas PE → Unix (syscalls NT y unix calls). Solo se sustituye
  `dlls/ntdll/unix/signal_x86_64.c` por un backend C que conduce `VpCpu` (`signal_vpk.c`).
- Steam = cliente real (él gestiona propiedad y DRM). Implica multiproceso y Chromium/CEF.
- Descartado: Win32 propio (`runtime/win/vp_win.h`, solo cabecera): no llega a Steam ni a launchers.
- No evaluado a fondo: Wine ARM64EC (lo que usan CrossOver ARM64 y Madeira con FEX). Sus DLL son
  código ARM64 cargado desde PE en ejecución: sin JIT harían falta gemelos Mach-O firmados.

## Medido (2026-11-11)
- `vpaot` sobre los 768 PE x86-64 de Wine 11.0 (build `wine-11.0-amd64` de Kron4ek, solo como
  datos): 288 192 funciones, 14 732 050 instrucciones, 1 597 sin soporte (99,989 %), 384 303
  llamadas indirectas, 35 075 saltos indirectos, 4 975 tablas de salto, 1 822 `syscall`. Traducir
  todo: 20 s en 2 núcleos. Compilar el C: no medido (`ntdll.dll` = 131 745 instrucciones = 32 MB
  de C con `--pic`; extrapolado, ~3,5 GB para todo Wine).
- Sin soporte: ud2 747, int1 215, in 142, out 111, cli 76, int3 39, insb 39, sti 38, hlt 31,
  outsb 23, mov a registro de segmento 22, iretd 20, outsd 18, insd 15, int 14, resto < 10. Son
  trampas, privilegiadas o datos decodificados como código. `vp_unsupported` ya hace `VP_OUT()`
  antes: el embebedor recibe `vp_run` = 3 con `fault_rip`/`fault_what` y registros precisos.
- Peores ficheros: winedbg.exe 340, glu32 156, dbghelp 138, user32 130, wined3d 102, ntdll 26.
- Leer un registro de segmento ya se traduce (RtlCaptureContext lo usa en cada excepción): PE →
  cs 0x33, ss/ds/es/gs 0x2b, fs 0x53. Test `tests/aot/cases/mov_seg.s`.

## Costura PE ↔ Unix en Wine 11.0 (leído del código; rutas del árbol de Wine)
- Thunk de syscall (`include/wine/asm.h:287-301`): `mov %rcx,%r10; mov $id,%eax; testb $1,0x7ffe0308;
  jne; syscall; ret; ...; call *(0x7ffe1000); ret`. `0x7ffe0308` vale 1 (`unix/virtual.c:4450`):
  siempre va por `call *(0x7ffe1000)`, página donde `signal_init_process` escribe
  `__wine_syscall_dispatcher` (`unix/signal_x86_64.c:2843-2846`).
- Dispatcher (`signal_x86_64.c:3026-3327`): tabla = bits 12-13 del id sobre `teb->syscall_table`
  (`KeServiceDescriptorTable[4]`, `unix/loader.c:141-158`), número = id & 0xfff, límite →
  0xc000001c; `ArgumentTable[n]` = bytes de argumentos; args = r10, rdx, r8, r9 y la pila desde
  rsp+0x28; resultado en rax. Guarda el estado de usuario en `struct syscall_frame` (0x300 bytes,
  `signal_x86_64.c:462-495`) colgado de `%gs:0x378`.
- Unix calls: el PE llama a `__wine_unix_call_dispatcher` (rcx = tabla, edx = código, r8 = args) →
  `NTSTATUS (*unixlib_entry_t)(void*)` (`include/wine/unixlib.h:34`, `signal_x86_64.c:3352-3464`).
  La tabla sale de `NtQueryVirtualMemory(MemoryWineUnixFuncs)` → dlopen perezoso de
  `<so_dir>/x86_64-unix/<dll>.so` + `__wine_unix_call_funcs` (`unix/virtual.c:837-863`).
- Arranque: `__wine_main` (`unix/loader.c:2278`) → `start_main_thread` (1908) →
  `server_init_process_done` (`unix/server.c:1731`) → `signal_start_thread`. `init_syscall_frame`
  (`signal_x86_64.c:2896-2976`) deja la primera instrucción PE en `LdrInitializeThunk` con
  rcx = CONTEXT* (Rcx = entry, Rdx = arg, Rip = `RtlUserThreadStart`, EFlags 0x200, MxCsr 0x1f80),
  rdx = arg, rsp = ctx - 8. Igual para todos los hilos (`unix/thread.c:1121-1132`).
- Unix → PE: `KeUserModeCallback` (1905) monta `callback_stack_layout` y entra en
  `KiUserCallbackDispatcher`; vuelve con `NtCallbackReturn` (1927). APC:
  `call_user_apc_dispatcher` (1661). Excepciones: `call_user_exception_dispatcher` (1720),
  `signal_set_full_context` (1083, lo usa `NtContinueEx`), `NtGet/SetContextThread` (1215/1116).
- Señales (`signal_x86_64.c`): `segv_handler` 2394 (`virtual_handle_fault`, guardas de pila),
  `usr1_handler` 2622 (suspensión), `quit_handler` 2608, `fpe`/`trap`/`int`/`abrt`.
- Mapeo de imágenes: núcleo `virtual_map_image` (`unix/virtual.c:3378`); bloque de éxito 3442-3448
  = punto único para "imagen mapeada" (base, tamaño, nombre NT). Desmapeo: `delete_view` 1790.
- Páginas: Windows fijo a 4K (`page_size`, `pages_vprot`). Página host variable solo bajo
  `__aarch64__` (`unix/virtual.c:176-181`); USD en 0x7ffe0000 y la página 0x7ffe1000 son de 4K.
- Procesos: `NtCreateUserProcess` (`unix/process.c:682`) = socketpair + doble fork + exec del
  loader con `WINESERVERSOCKET`; wineserver es otro proceso (`unix/loader.c:572-599`). Estado
  global de un solo proceso Windows por proceso Unix: `peb`, `main_image_info`, `fd_cache`,
  `views_tree`, `pages_vprot`, `teb_list`, `KeServiceDescriptorTable`, punteros `pKiUser*`.
- Sin verificar: `#ifdef __x86_64__` fuera de `signal_x86_64.c` (server/, win32u), manejo de x18 y
  excepciones mach en `signal_arm64.c`, lado Unix de win32u.

## Backend `signal_vpk.c` (plan)
- Estado de usuario = un `VpCpu` por hilo; `syscall_frame` se mantiene como struct C sincronizada
  con él para que contexto, APC, excepciones y servidor no cambien.
- `0x7ffe1000` y `__wine_unix_call_dispatcher` pasan a ser direcciones invitadas registradas con
  `vp_register_native`; el `syscall` crudo entra por `VpEmbedderHooks.syscall`.
- `signal_start_thread` = rellenar `VpCpu` y `vp_run`. `KeUserModeCallback` = `vp_run` anidado;
  `NtCallbackReturn` pone `rip = VP_HOST_EXIT_ADDRESS`. `NtContinue`, APC y excepciones = escribir
  `VpCpu` y seguir (destinos a mitad de función: entradas extra o traducción bajo demanda).
- Trampas (`vp_run` = 3) → excepción NT según `fault_what`.
- Abierto: fallo de memoria visible para el invitado (el C traducido tiene los registros en
  locales: no hay rip/registros precisos en un SIGSEGV); suspensión de otro hilo fuera de un punto
  seguro.
- Módulos: `vpaot --pic --module` por DLL; `vp_attach_module(base, size)` en el bloque de éxito de
  `virtual_map_image`, `vp_detach_module` en `delete_view`. DLL de Wine traducidas en CI; exe y DLL
  del usuario, en el visor (`vpconvert`).

## Lo que exige el objetivo además del backend
1. Varios procesos Windows en un proceso (Steam + `steamwebhelper`, launcher → juego): wineserver
   como hilo; una copia de las bibliotecas Unix de Wine por proceso Windows; shim POSIX por
   proceso (fds, mmap, hilos, spawn); reparto del espacio de direcciones; fin de proceso
   cooperativo en la costura. Diseño sin verificar.
2. Código que no existe al traducir (V8 de CEF, .NET, Mono de Unity, imagen cifrada por el
   envoltorio de Steam): traducción tardía tras escribirse la imagen, intérprete x86-64 de
   respaldo (no existe) y arrancar Chromium sin JIT (`--js-flags=--jitless`, sin sandbox; sin
   verificar que Steam lo admita).
3. arm64/visionOS: lado Unix arm64 con invitado x86-64 (Wine ata la arquitectura PE a la del
   host), páginas de 16K, firma y carga de las traducciones.
4. Driver de ventanas, entrada y audio para visionOS; fuentes.
5. Gráficos: winevulkan → VPVulkan; DXVK y vkd3d-proton traducidos; después DXBC/DXIL → VPIR.
6. x86 de 32 bits (`vpaot` + WoW64): el cliente de Steam es de 64 bits desde 2025-12; muchos
   juegos y launchers no.
7. Fuera de alcance: anti-cheat de kernel.

## Hitos (cada uno con su comprobación)
- M1 Linux x86-64: `hello.exe` de consola con ntdll/kernelbase/kernel32/ucrtbase de Wine
  traducidos, `ntdll.so` con `signal_vpk.c` y wineserver normal. Comprobación: salida y código de
  salida; ninguna instrucción PE ejecutada de forma nativa.
- M2: tests de conformidad de Wine (ntdll, kernel32) sobre el backend: SEH, C++ EH, hilos, APC,
  callbacks.
- M3: traducción tardía + intérprete; un proceso Chromium sin JIT.
- M4: contenedor multiproceso en Linux: `CreateProcess` padre → hijo en el mismo proceso.
- M5: arm64 Darwin/visionOS: consola en el visor.
- M6: ventanas, entrada, audio; Steam instala, inicia sesión y descarga.
- M7: Vulkan/D3D; primer juego D3D11 de Steam.
- M8: 32 bits, launchers, rendimiento.

## Entorno
- Wine 11.0: `git clone --depth 1 --branch wine-11.0 https://github.com/wine-mirror/wine`.
- Compilar los PE de Wine necesita flex (no instalado aquí) y un cruzado PE (clang 18 + lld-link
  sí están). Compilar flex o Wine en este entorno requiere permiso expreso; alternativa: CI.
