// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Converting a game on the headset (tools/vpconvert): its x86-64 code translated, compiled and
// linked into a game pack by the app itself, then signed and loaded (VPGamePack.swift). No PC.
//
// Slow (minutes to tens of minutes for a large game), so it is resumable: the state lives in
// Library/Application Support/VPEngine/Conversion/<game>, and a run that is stopped continues
// where it was. It runs while the app is open; when the app goes to the background it asks for
// the usual short extension, and it schedules a background processing task (BGProcessingTask:
// visionOS gives the app time when the headset is idle, mostly while charging), each run of which
// continues the conversion until the system ends it.
//
// Everything that happens goes to Documents/conversion.log (visible in the Files app): each run,
// each module and piece with its time, a sample of memory and temperature every 15 s, whether the
// app was in the foreground, and the totals.

import BackgroundTasks
import Foundation
import Observation
import UIKit
import os

@MainActor
@Observable
final class VPConversion {
    static let shared = VPConversion()
    /// Info.plist: BGTaskSchedulerPermittedIdentifiers holds it, UIBackgroundModes "processing".
    static let taskIdentifier = "vpengine.conversion"

    enum State: Equatable {
        case idle
        case running(phase: Int32, done: Int, total: Int)
        case paused
        case failed(String)
        case finished(VPGamePack.Loaded)
    }

    private(set) var state: State = .idle
    /// What the conversion is doing now, in words, for the launcher.
    private(set) var status = ""
    /// When the current run started and how much of the whole conversion was done before it.
    private(set) var runStarted: Date?

    /// The game folder to convert when a background task runs (the app knows its VPS4 folder).
    var resolveGame: () -> URL? = { nil }
    /// Called when a conversion finished and its pack is loaded.
    var onFinished: (VPGamePack.Loaded) -> Void = { _ in }

    /// Pieces compiled at once (Settings). More is faster until memory or heat get in the way.
    var jobs: Int {
        get { max(1, min(8, UserDefaults.standard.object(forKey: "vpengine.jobs") as? Int ?? 4)) }
        set { UserDefaults.standard.set(max(1, min(8, newValue)), forKey: "vpengine.jobs") }
    }

    var isRunning: Bool {
        if case .running = state { return true }
        return false
    }

    private let stop = StopFlag()
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var processingTask: BGProcessingTask?
    private var sampler: Task<Void, Never>?
    private var inForeground = true

    // MARK: - Where things are

    nonisolated static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VPEngine/Conversion", isDirectory: true)
    }

    nonisolated static func workDirectory(for game: URL) -> URL {
        root.appendingPathComponent(game.lastPathComponent, isDirectory: true)
    }

    nonisolated static func packURL(for game: URL) -> URL {
        workDirectory(for: game).appendingPathComponent(VPGamePack.fileName)
    }

    /// The game's files as they are now: a pack converted from other files is not this game's.
    nonisolated static func gameStamp(_ game: URL) -> String {
        let manager = FileManager.default
        var files = [game.appendingPathComponent("eboot.bin")]
        let modules = game.appendingPathComponent("sce_module")
        files += ((try? manager.contentsOfDirectory(at: modules, includingPropertiesForKeys: nil)) ?? [])
            .filter { ["prx", "sprx"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return files.map { url -> String in
            let a = try? manager.attributesOfItem(atPath: url.path)
            let size = (a?[.size] as? NSNumber)?.int64Value ?? -1
            let date = (a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            return "\(url.lastPathComponent) \(size) \(Int64(date))"
        }.joined(separator: "\n")
    }

    /// Whether the game was converted (and its files did not change since).
    nonisolated static func isConverted(_ game: URL) -> Bool {
        let done = workDirectory(for: game).appendingPathComponent("done")
        guard let stamp = try? String(contentsOf: done, encoding: .utf8) else { return false }
        return stamp == gameStamp(game) && FileManager.default.fileExists(atPath: packURL(for: game).path)
    }

    /// Whether a conversion was started and not finished (it continues from there).
    nonisolated static func isStarted(_ game: URL) -> Bool {
        FileManager.default.fileExists(atPath: workDirectory(for: game).path) && !isConverted(game)
    }

    /// Throws the conversion away (to start over).
    nonisolated static func reset(_ game: URL) {
        try? FileManager.default.removeItem(at: workDirectory(for: game))
    }

    // MARK: - Running

    /// Converts `game` (or continues converting it). `reason` goes to the log.
    func start(game: URL, reason: String) {
        if isRunning { return }
        if Self.isConverted(game) {
            loadConverted(game, reason: reason)
            return
        }
        guard let sdk = Bundle.main.url(forResource: "VPEngineSDK", withExtension: nil) else {
            state = .failed(L("Falta VPEngineSDK en la app (build incompleta).", "VPEngineSDK is missing from the app (incomplete build)."))
            return
        }
        do {
            _ = try VPCertificate.storedIdentity()
        } catch {
            state = .failed(error.localizedDescription)
            return
        }
        stop.set(false)
        let jobs = self.jobs
        let work = Self.workDirectory(for: game)
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        UserDefaults.standard.set(game.path, forKey: "vpengine.conversion.game")
        let log = ConversionLog.shared
        let previous = Timing.load(work)
        log.beginRun(reason: reason, game: game, jobs: jobs, previous: previous, foreground: inForeground)
        runStarted = Date()
        state = .running(phase: 0, done: 0, total: 0)
        status = L("Traduciendo…", "Translating…")
        startSampler()

        let context = Context(owner: self, log: log, stop: stop)
        let unmanaged = Unmanaged.passRetained(context)
        let thread = Thread { [weak self] in
            var config = VpConvertConfig()
            let result: Int32 = game.path.withCString { gamePath in
                work.path.withCString { workPath in
                    Self.packURL(for: game).path.withCString { outPath in
                        sdk.path.withCString { sdkPath in
                            game.lastPathComponent.withCString { title in
                                config.game_dir = gamePath
                                config.work_dir = workPath
                                config.output = outPath
                                config.sdk_dir = sdkPath
                                config.title = title
                                config.jobs = Int32(jobs)
                                let missing = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                                    .appendingPathComponent("vpengine_missing.txt")
                                return (FileManager.default.fileExists(atPath: missing.path) ? missing.path : "").withCString { missingPath in
                                    config.missing_log = missingPath.pointee == 0 ? nil : missingPath
                                    var callbacks = VpConvertCallbacks()
                                    callbacks.user = unmanaged.toOpaque()
                                    callbacks.log = { user, line in
                                        guard let user, let line else { return }
                                        Unmanaged<Context>.fromOpaque(user).takeUnretainedValue().log.line(String(cString: line))
                                    }
                                    callbacks.progress = { user, phase, done, total in
                                        guard let user else { return }
                                        let context = Unmanaged<Context>.fromOpaque(user).takeUnretainedValue()
                                        context.progress(phase: phase, done: Int(done), total: Int(total))
                                    }
                                    callbacks.should_stop = { user in
                                        guard let user else { return 1 }
                                        return Unmanaged<Context>.fromOpaque(user).takeUnretainedValue().stop.get() ? 1 : 0
                                    }
                                    return vp_convert(&config, &callbacks)
                                }
                            }
                        }
                    }
                }
            }
            unmanaged.release()
            // Converted: sign it and load it here, off the main thread (a few seconds).
            var loaded: VPGamePack.Loaded?
            var failure: String?
            if result == 0 {
                do {
                    try Self.gameStamp(game).write(to: work.appendingPathComponent("done"), atomically: true, encoding: .utf8)
                    let started = Date()
                    loaded = try VPGamePack.load(pack: Self.packURL(for: game))
                    log.line(String(format: "signed and loaded the pack in %.1f s", Date().timeIntervalSince(started)))
                } catch {
                    failure = error.localizedDescription
                    log.line("ERROR: \(error.localizedDescription)")
                }
            }
            Task { @MainActor [loaded, failure] in
                self?.runEnded(result: result, game: game, work: work, loaded: loaded, failure: failure)
            }
        }
        thread.stackSize = 64 << 20 // vpaot and lld run on it
        thread.qualityOfService = .userInitiated
        thread.name = "VPEngine conversion"
        thread.start()
    }

    /// Ends the run after the pieces being compiled (they take seconds). The rest waits.
    func requestStop(reason: String) {
        guard isRunning else { return }
        ConversionLog.shared.line("stop requested: \(reason)")
        stop.set(true)
        status = L("Parando…", "Stopping…")
    }

    private func runEnded(result: Int32, game: URL, work: URL, loaded: VPGamePack.Loaded?, failure: String?) {
        sampler?.cancel()
        sampler = nil
        let seconds = runStarted.map { Date().timeIntervalSince($0) } ?? 0
        var timing = Timing.load(work)
        timing.runs += 1
        timing.activeSeconds += seconds
        if timing.firstStart == 0 { timing.firstStart = runStarted?.timeIntervalSince1970 ?? Date().timeIntervalSince1970 }
        timing.save(work)
        let log = ConversionLog.shared
        switch result {
        case 0 where loaded != nil:
            state = .finished(loaded!)
            status = L("Convertido", "Converted")
            log.endRun(result: "FINISHED", seconds: seconds, timing: timing, finished: true)
            onFinished(loaded!)
        case 0:
            state = .failed(failure ?? "?")
            log.endRun(result: "converted, but the pack did not load", seconds: seconds, timing: timing, finished: false)
        case 1:
            state = .paused
            status = L("En pausa: seguirá donde iba", "Paused: it will continue where it was")
            log.endRun(result: "paused (resumable)", seconds: seconds, timing: timing, finished: false)
            scheduleBackgroundContinuation()
        default:
            state = .failed(L("La conversión falló: mira conversion.log en Archivos.", "The conversion failed: see conversion.log in Files."))
            log.endRun(result: "FAILED (\(result))", seconds: seconds, timing: timing, finished: false)
        }
        runStarted = nil
        processingTask?.setTaskCompleted(success: result == 0)
        processingTask = nil
        endBackgroundTime()
    }

    private func loadConverted(_ game: URL, reason: String) {
        status = L("Cargando el juego convertido…", "Loading the converted game…")
        Thread.detachNewThread { [weak self] in
            let result = Result { try VPGamePack.load(pack: Self.packURL(for: game)) }
            Task { @MainActor in
                switch result {
                case .success(let loaded):
                    self?.state = .finished(loaded)
                    self?.status = L("Convertido", "Converted")
                    self?.onFinished(loaded)
                case .failure(let error):
                    self?.state = .failed(error.localizedDescription)
                    ConversionLog.shared.line("loading the converted pack failed: \(error.localizedDescription)")
                }
            }
        }
    }

    fileprivate func progressed(phase: Int32, done: Int, total: Int) {
        state = .running(phase: phase, done: done, total: total)
        switch phase {
        case 0: status = L("Traduciendo módulos: \(done) de \(total)", "Translating modules: \(done) of \(total)")
        case 1: status = L("Compilando: \(done) de \(total) piezas", "Compiling: \(done) of \(total) pieces")
        default: status = L("Enlazando…", "Linking…")
        }
    }

    // MARK: - Foreground and background

    /// The app's scene phase changed (the app calls this).
    func appBecameActive() {
        inForeground = true
        ConversionLog.shared.lineIfRunning(isRunning, "app in the foreground")
        if !isRunning, let path = UserDefaults.standard.string(forKey: "vpengine.conversion.game"),
           case .paused = state, let game = resolveGame(), game.path == path {
            start(game: game, reason: "back in the foreground")
        }
    }

    func appEnteredBackground() {
        inForeground = false
        guard isRunning else { return }
        ConversionLog.shared.line("app in the background: asking for time to go on")
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "VPEngine conversion") { [weak self] in
            Task { @MainActor in
                self?.requestStop(reason: "the background time ran out")
            }
        }
        scheduleBackgroundContinuation()
    }

    private func endBackgroundTime() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    /// Before the app finishes launching: what runs when visionOS grants a background task.
    static func registerBackgroundTask() {
        let registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: taskIdentifier, using: .main) { task in
            guard let task = task as? BGProcessingTask else { task.setTaskCompleted(success: false); return }
            MainActor.assumeIsolated {
                let me = VPConversion.shared
                ConversionLog.shared.line("background task granted by the system")
                guard let game = me.resolveGame(), !VPConversion.isConverted(game) else {
                    task.setTaskCompleted(success: true)
                    return
                }
                me.processingTask = task
                task.expirationHandler = {
                    Task { @MainActor in VPConversion.shared.requestStop(reason: "the background task expired") }
                }
                if me.isRunning {
                    ConversionLog.shared.line("background task: the conversion is already running")
                } else {
                    me.start(game: game, reason: "background task")
                }
            }
        }
        if !registered {
            ConversionLog.shared.line("background task could not be registered (Info.plist BGTaskSchedulerPermittedIdentifiers?)")
        }
    }

    func scheduleBackgroundContinuation() {
        let request = BGProcessingTaskRequest(identifier: Self.taskIdentifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        do {
            try BGTaskScheduler.shared.submit(request)
            ConversionLog.shared.line("background task requested (visionOS decides when: usually idle, charging)")
        } catch {
            ConversionLog.shared.line("background task could not be requested: \(error.localizedDescription)")
        }
    }

    // MARK: - Samples

    private func startSampler() {
        sampler?.cancel()
        sampler = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(15))
                guard let self, self.isRunning else { return }
                ConversionLog.shared.sample(state: self.state, foreground: self.inForeground)
            }
        }
    }

    /// What the C callbacks reach (any thread).
    private final class Context: @unchecked Sendable {
        weak var owner: VPConversion?
        let log: ConversionLog
        let stop: StopFlag
        private let lock = NSLock()
        private var lastPost = Date.distantPast
        init(owner: VPConversion, log: ConversionLog, stop: StopFlag) {
            self.owner = owner
            self.log = log
            self.stop = stop
        }
        func progress(phase: Int32, done: Int, total: Int) {
            lock.lock()
            let now = Date()
            let post = done == total || now.timeIntervalSince(lastPost) > 0.25
            if post { lastPost = now }
            lock.unlock()
            guard post else { return }
            Task { @MainActor [weak owner] in owner?.progressed(phase: phase, done: done, total: total) }
        }
    }
}

/// A flag the conversion threads read while the main thread sets it.
final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
    func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
}

/// Totals across the runs of one conversion (kept in its folder).
struct Timing: Codable {
    var runs = 0
    var activeSeconds: Double = 0
    var firstStart: Double = 0

    static func load(_ work: URL) -> Timing {
        guard let data = try? Data(contentsOf: work.appendingPathComponent("timing.json")),
              let t = try? JSONDecoder().decode(Timing.self, from: data) else { return Timing() }
        return t
    }

    func save(_ work: URL) {
        if let data = try? JSONEncoder().encode(self) {
            try? data.write(to: work.appendingPathComponent("timing.json"))
        }
    }
}

/// Documents/conversion.log: appended to by every run, with wall-clock time and time into the run.
final class ConversionLog: @unchecked Sendable {
    static let shared = ConversionLog()
    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("conversion.log")
    private let queue = DispatchQueue(label: "vpengine.conversion.log")
    private var runStart = Date()
    private let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    func line(_ text: String) {
        let now = Date()
        queue.async { [self] in
            let elapsed = now.timeIntervalSince(runStart)
            let stamp = String(format: "%@ +%02d:%04.1f ", formatter.string(from: now), Int(elapsed) / 60, elapsed.truncatingRemainder(dividingBy: 60))
            append(text.split(separator: "\n", omittingEmptySubsequences: false).map { stamp + $0 }.joined(separator: "\n") + "\n")
        }
        VPGamePack.log("conversion: " + text)
    }

    func lineIfRunning(_ running: Bool, _ text: String) {
        if running { line(text) }
    }

    func beginRun(reason: String, game: URL, jobs: Int, previous: Timing, foreground: Bool) {
        queue.sync { runStart = Date() }
        let info = ProcessInfo.processInfo
        let free = (try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        line("=== conversion run \(previous.runs + 1) of \(game.lastPathComponent) (\(reason)) ===")
        line("device \(Self.machine()), \(info.activeProcessorCount) cores, \(info.physicalMemory >> 20) MB RAM, \(free >> 20) MB free on disk, \(jobs) pieces at once, \(foreground ? "foreground" : "background"), thermal \(Self.thermal())")
        if previous.runs > 0 {
            line(String(format: "before this run: %d runs, %.1f min of work", previous.runs, previous.activeSeconds / 60))
        }
    }

    func endRun(result: String, seconds: Double, timing: Timing, finished: Bool) {
        line(String(format: "run ended: %@ after %.1f min", result, seconds / 60))
        if finished {
            let wall = Date().timeIntervalSince1970 - timing.firstStart
            line(String(format: "TOTAL: converted in %.1f min of work over %d runs (%.1f min from the first start to the end)",
                        timing.activeSeconds / 60, timing.runs, wall / 60))
        }
    }

    @MainActor
    func sample(state: VPConversion.State, foreground: Bool) {
        var progress = ""
        if case .running(let phase, let done, let total) = state {
            progress = "phase \(["translate", "compile", "link"][Int(max(0, min(2, phase)))]) \(done)/\(total)"
        }
        line("sample: \(progress), memory \(Self.footprint() >> 20) MB used, \(os_proc_available_memory() >> 20) MB more allowed, thermal \(Self.thermal()), \(foreground ? "foreground" : "background")")
    }

    private func append(_ text: String) {
        let data = Data(text.utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "SERIOUS"
        case .critical: return "CRITICAL"
        @unknown default: return "?"
        }
    }

    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let r = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return r == KERN_SUCCESS ? info.phys_footprint : 0
    }

    static func machine() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        return String(cString: buf)
    }
}
