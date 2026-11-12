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
    private var startingSet: Set<ObjectIdentifier> = [] // what the start under way is for
    private var pending: [GCController]? // the newest set asked for while a start was under way
    private var epoch = 0 // raised by every halt (a stop(), or a start that replaces the tracking): a start that was under way when it came must not keep its session
    private var startingEpoch = 0 // the epoch the start under way began in (set under the lock, with `starting`)

    private init() {}

    /// Tracks these controllers (and no others). Returns at once; loading the accessories and
    /// asking for the permission go on in the background. Calling it again with the same
    /// controllers does nothing. Asked while a start is under way, it is done when that one ends
    /// (the newest set asked for wins).
    func track(_ controllers: [GCController]) {
        let wanted = Set(controllers.map(ObjectIdentifier.init))
        lock.lock()
        if starting {
            // The start under way is the one asked for only if no stop() has come since it began.
            pending = (wanted == startingSet && epoch == startingEpoch) ? nil : controllers
            lock.unlock()
            return
        }
        if wanted == tracked {
            lock.unlock()
            return
        }
        starting = true
        startingSet = wanted
        let old = takeAway()
        let began = epoch // after takeAway, in the same lock hold: a stop() from here on makes the start stale
        startingEpoch = began
        lock.unlock()
        release(old)
        guard !controllers.isEmpty else {
            endStart()
            return
        }
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.start(controllers, wanted: wanted, began: began)
        }
    }

    /// Ends the start under way, and begins the set asked for meanwhile, if there is one.
    private func endStart() {
        let next = lock.withLock { () -> [GCController]? in
            starting = false
            startingSet = []
            let next = pending
            pending = nil
            return next
        }
        if let next { track(next) }
    }

    private func start(_ controllers: [GCController], wanted: Set<ObjectIdentifier>, began: Int) async {
        defer { endStart() }
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
        // The session and its updates are kept together, under the lock that checks that no stop()
        // came since the start began: a stop() can never find the session without its task.
        let stale = lock.withLock { () -> Bool in
            if epoch != began { return true }
            self.session = session
            tracked = wanted
            task = Task.detached(priority: .userInitiated) {
                for await update in provider.anchorUpdates {
                    if Task.isCancelled { break }
                    Self.send(update.anchor)
                }
            }
            return false
        }
        if stale {
            session.stop()
            return
        }
        LogFiles.log("Sense tracking: on for \(accessories.count) controller(s)")
    }

    func stop() {
        lock.withLock { pending = nil } // nothing asked for before the stop is wanted after it
        halt()
    }

    /// Stops the session and the updates; what was asked for during a start (pending) stays.
    private func halt() {
        release(lock.withLock { takeAway() })
    }

    /// Takes the session and the updates away and raises the epoch. The lock is held.
    private func takeAway() -> (ARKitSession?, Task<Void, Never>?) {
        let taken = (session, task)
        session = nil
        task = nil
        tracked = []
        epoch += 1
        return taken
    }

    /// Stops what takeAway took, and lets go of the aim of both hands. Not under the lock.
    private func release(_ taken: (ARKitSession?, Task<Void, Never>?)) {
        taken.1?.cancel()
        taken.0?.stop()
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
