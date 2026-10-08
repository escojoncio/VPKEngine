// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// PlayStation VR2 Sense controllers tracked in space by the headset (ARKit's accessory tracking,
// visionOS 26): each one's position and orientation go to the core (VPPlatform.aimSink) as the pose
// of the hand holding it. The game holds the flashlight with it when the settings say so, and
// points at its menus with it (the trigger clicks).

import ARKit
import Foundation
import GameController
import simd

final class SenseTracking: @unchecked Sendable {
    static let shared = SenseTracking()

    private let lock = NSLock()
    private var session: ARKitSession?
    private var task: Task<Void, Never>?
    private var tracked: Set<ObjectIdentifier> = []
    private var starting = false

    private init() {}

    /// Tracks these controllers (and no others). Returns at once; loading the accessories and
    /// asking for the permission go on in the background. Calling it again with the same
    /// controllers does nothing.
    func track(_ controllers: [GCController]) {
        let wanted = Set(controllers.map(ObjectIdentifier.init))
        lock.lock()
        if wanted == tracked || starting {
            lock.unlock()
            return
        }
        starting = true
        lock.unlock()
        stop()
        guard !controllers.isEmpty else {
            lock.lock()
            starting = false
            lock.unlock()
            return
        }
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.start(controllers, wanted: wanted)
        }
    }

    private func start(_ controllers: [GCController], wanted: Set<ObjectIdentifier>) async {
        defer {
            lock.withLock { starting = false }
        }
        guard AccessoryTrackingProvider.isSupported else {
            LogFiles.log("Sense tracking: not supported here")
            return
        }
        var accessories: [Accessory] = []
        for controller in controllers {
            do {
                accessories.append(try await Accessory(device: controller))
            } catch {
                LogFiles.log("Sense tracking: no accessory for \(controller.vendorName ?? "a controller") (\(error.localizedDescription))")
            }
        }
        guard !accessories.isEmpty else { return }
        let session = ARKitSession()
        _ = await session.requestAuthorization(for: AccessoryTrackingProvider.requiredAuthorizations)
        let provider = AccessoryTrackingProvider(accessories: accessories)
        do {
            try await session.run([provider])
        } catch {
            LogFiles.log("Sense tracking: could not start (\(error.localizedDescription))")
            return
        }
        lock.withLock {
            self.session = session
            tracked = wanted
        }
        LogFiles.log("Sense tracking: on for \(accessories.count) controller(s)")
        let updates = Task.detached(priority: .userInitiated) {
            for await update in provider.anchorUpdates {
                if Task.isCancelled { break }
                Self.send(update.anchor)
            }
        }
        lock.withLock { task = updates }
    }

    func stop() {
        lock.lock()
        let session = self.session
        let task = self.task
        self.session = nil
        self.task = nil
        tracked = []
        lock.unlock()
        task?.cancel()
        session?.stop()
        for hand in 0...1 {
            VPPlatform.aimSink(hand, false, .zero, SIMD4(0, 0, 0, 1))
        }
    }

    private static func send(_ anchor: AccessoryAnchor) {
        // The hand holding it, else the hand it is made for.
        let held = String(describing: anchor.heldChirality).lowercased()
        let inherent = "\(anchor.accessory.inherentChirality)".lowercased()
        let isLeft = held.contains("left") || (!held.contains("right") && inherent.contains("left"))
        let hand = isLeft ? 0 : 1
        guard anchor.isTracked else {
            VPPlatform.aimSink(hand, false, .zero, SIMD4(0, 0, 0, 1))
            return
        }
        let m = anchor.originFromAnchorTransform
        let rotation = simd_quatf(simd_float3x3(
            SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
        VPPlatform.aimSink(hand, true, SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z), rotation.vector)
    }
}
