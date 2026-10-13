// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// The one place a game app configures the shared visionOS layer: where the app's files are, the
// game core's log path, where controller states and tracked-controller poses go, and the headset
// model. Everything else in this folder reads these and nothing of the game.

import Foundation

public enum VPHeadset {
    case m2 // Apple Vision Pro (2024), RealityDevice14,x
    case m5 // a later headset
}

/// A controller's state, as the games take it (mirrors `vp_controller` in vp_platform.h).
public struct VPControllerState: Equatable {
    public var active = false              // a controller is connected and this state is current
    public var moveX: Float = 0, moveY: Float = 0   // left stick, -1..1
    public var turnX: Float = 0, turnY: Float = 0   // right stick, -1..1
    public var cross = false, circle = false, square = false, triangle = false
    public var options = false             // OPTIONS
    public var touchpad = false            // touchpad / Create / Share
    public var l1 = false, r1 = false
    public var l2: Float = 0, r2: Float = 0 // triggers, 0..1
    public var l3 = false, r3 = false
    public var dpadUp = false, dpadDown = false, dpadLeft = false, dpadRight = false
    public var handValid = [false, false]
    public var handPosition: [SIMD3<Float>] = [.zero, .zero]
    public var handRotation: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0, 1)] // quaternion x,y,z,w
    public var promptStyle = 1             // button glyphs: 0 Xbox, 1 PlayStation, 2 Nintendo
    public init() {}
}

public enum VPPlatform {
    /// The app's Documents folder (also reachable from the Files app).
    public static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// The game core's own log file, once the core started; set by the game app.
    public static var coreLogPath: () -> String? = { nil }

    /// Where controller states go (the game app hands them to its core).
    public static var controllerSink: (VPControllerState) -> Void = { _ in }

    /// Where tracked-controller poses go: hand 0 left, 1 right; position in metres and
    /// orientation (x, y, z, w) in the immersive space, the controller pointing along its -Z.
    public static var aimSink: (_ hand: Int, _ valid: Bool, _ position: SIMD3<Float>, _ orientation: SIMD4<Float>) -> Void = { _, _, _, _ in }

    /// The os.Logger subsystem of the app's own lines.
    public static var logSubsystem = "vpkengine"

    /// Which headset this is: the first Apple Vision Pro (M2) is RealityDevice14,x; a later one
    /// is taken as an M5. Unknown: M2, the safer preset.
    public static func detectedHeadset() -> VPHeadset {
        let model = hardwareModel()
        if model.contains("RealityDevice14") { return .m2 }
        if model.contains("RealityDevice") { return .m5 }
        if ProcessInfo.processInfo.physicalMemory >= 24 * 1024 * 1024 * 1024 { return .m5 }
        return .m2
    }

    /// "RealityDevice14,1" and the like, from sysctl (hw.machine, else hw.model).
    public static func hardwareModel() -> String {
        for name in ["hw.machine", "hw.model"] {
            var size = 0
            guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { continue }
            var buffer = [CChar](repeating: 0, count: size + 1)
            guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { continue }
            let value = String(cString: buffer)
            if !value.isEmpty { return value }
        }
        return ""
    }
}
