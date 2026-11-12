// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// The logs: the game core writes its own file (VPPlatform.coreLogPath, inside the app's Documents, so
// it is also reachable from the Files app), and the launcher shows its last lines and sends it.
// The app's own lines go to the system log (Console) with a prefix, and to a small file of their
// own in Documents/Registros, so a crash of the core still leaves what the app did before it.

import Foundation
import os

enum LogFiles {
    private static let logger = Logger(subsystem: VPPlatform.logSubsystem, category: "app")
    private static let lock = NSLock()
    private static var appLogHandle: FileHandle?

    static var directory: URL {
        VPPlatform.documents.appendingPathComponent("Registros", isDirectory: true)
    }

    /// The app's own log of this session.
    static var appLogURL: URL {
        directory.appendingPathComponent("app.txt")
    }

    /// The core's log, once it has started.
    static var coreLogURL: URL? {
        guard let path = VPPlatform.coreLogPath(), !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// At the app's start: a fresh app log.
    static func begin() {
        let manager = FileManager.default
        try? manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try? manager.removeItem(at: appLogURL)
        manager.createFile(atPath: appLogURL.path, contents: nil)
        lock.lock()
        appLogHandle = try? FileHandle(forWritingTo: appLogURL)
        lock.unlock()
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        log("P.T. VR \(version) (\(Bundle.main.bundleIdentifier ?? "?")), "
            + "\(ProcessInfo.processInfo.operatingSystemVersionString), "
            + "\(VPPlatform.hardwareModel()), \(ProcessInfo.processInfo.physicalMemory / (1024 * 1024)) MB")
    }

    /// A line of the app's own.
    static func log(_ message: String) {
        logger.notice("\(message, privacy: .public)")
        let line = "\(stamp(Date())) \(message)\n"
        lock.lock()
        defer { lock.unlock() }
        if let data = line.data(using: .utf8) {
            // write(contentsOf:) throws; write(_:) raises an Objective-C exception (disk full) that ends the app.
            try? appLogHandle?.write(contentsOf: data)
        }
    }

    /// The last lines of a file (the whole file when it is small).
    static func tail(of url: URL?, lines wanted: Int = 300) -> String {
        guard let url, let handle = try? FileHandle(forReadingFrom: url) else {
            return ""
        }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let window: UInt64 = 256 * 1024
        let start = size > window ? size - window : 0
        try? handle.seek(toOffset: start)
        guard let data = try? handle.readToEnd() else {
            return ""
        }
        // The window can start inside a character: decoding with String(data:encoding:) would fail for the whole text.
        let text = String(decoding: data, as: UTF8.self)
        var lines = text.components(separatedBy: "\n")
        if start > 0, !lines.isEmpty {
            // The first line is probably cut.
            lines.removeFirst()
        }
        return lines.suffix(wanted).joined(separator: "\n")
    }

    /// The core's last error line, without its time and level ("" when there is none).
    static func lastCoreError() -> String {
        let lines = tail(of: coreLogURL, lines: 400).components(separatedBy: "\n")
        guard let line = lines.last(where: { $0.contains("] error ") }),
              let range = line.range(of: "] error ") else {
            return ""
        }
        let text = line[range.upperBound...].trimmingCharacters(in: .whitespaces)
        return text.count > 160 ? String(text.prefix(160)) + "…" : text
    }

    /// What is sent with the share button: the core's log and the app's.
    static func filesToShare() -> [URL] {
        var files: [URL] = []
        if let core = coreLogURL, FileManager.default.fileExists(atPath: core.path) {
            files.append(core)
        }
        if FileManager.default.fileExists(atPath: appLogURL.path) {
            files.append(appLogURL)
        }
        return files
    }

    // One formatter: building one per line costs more than the line.
    private static let stampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    private static func stamp(_ date: Date) -> String {
        stampFormatter.string(from: date)
    }
}
