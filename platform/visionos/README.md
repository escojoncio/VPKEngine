# Shared visionOS layer

Swift sources every VPEngine game app includes as they are (XcodeGen: add this folder's
`Sources` to the target), plus `vp_platform.h` for the C side:

| File | What it does |
|---|---|
| `VPPlatform.swift` | The configuration point: Documents folder, the core's log path, where controller states and tracked-controller poses go, headset model (M2 / later). |
| `VPS4Folder.swift` | The `VPS4` folder the player picks once (bookmark in defaults and keychain, survives reinstalls): `Juegos/`, `Partidas/`, `Cachés/`. |
| `PlayStationController.swift` | DualSense / DualShock 4 / generic gamepads and PlayStation VR2 Sense controllers → `VPControllerState`. |
| `SenseTracking.swift` | ARKit accessory tracking of the Sense controllers (visionOS 26) → `VPPlatform.aimSink`. |
| `LogFiles.swift` | The app's own log and the core's, in `Documents/Registros`. |
| `Language.swift` | Spanish / English / system, `L("…", "…")`. |

A game app sets the three hooks in `VPPlatform` at start and reads `VPS4Folder.url()` to find
its game. Nothing here knows which game it is.
