// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// The VPS4 folder: a folder the player makes once, called exactly "VPS4", at the top of
// "On My Apple Vision Pro" in the Files app, shared by the Vision Pro game ports. It lives
// outside the app, so the games and the saves in it survive deleting or replacing the app:
//
//   VPS4/Juegos/CUSA01127      the game's data (chunk1.psarc, texture.qar, ...)
//   VPS4/Partidas/CUSA01127    the saves
//
// An app cannot look outside its own folder by itself: the player picks VPS4 once (a folder
// picker that only accepts that name) and the app keeps a bookmark to it, in its settings and
// in the keychain (which outlives a reinstall; whether the bookmark still opens then is up to
// the system: if not, the player picks it again). The keys are AstroVisionPro's
// (visionos/App/Core/GameFolder.swift there): with the same bundle ID one pick serves both apps.

import Foundation
import Security

final class VPS4Folder: @unchecked Sendable {
    static let shared = VPS4Folder()

    static let folderName = "VPS4"
    static let gamesName = "Juegos"
    static let savesName = "Partidas"

    // Shared with AstroVisionPro's GameFolder: keep them the same.
    private static let defaultsKey = "vps4FolderBookmark"
    private static let keychainService = "astroquest.vps4"
    private static let keychainAccount = "folder-bookmark"

    private let lock = NSLock()
    private var opened: URL?

    private init() {}

    /// The folder, opened for this process (nil: not chosen yet, or the bookmark no longer opens).
    var url: URL? {
        lock.lock()
        defer { lock.unlock() }
        if let opened {
            return opened
        }
        // The app's settings first, then the keychain (which outlives a reinstall).
        let stored: [(String, Data?)] = [("settings", UserDefaults.standard.data(forKey: Self.defaultsKey)),
                                         ("keychain", Self.keychainBookmark())]
        for (source, data) in stored {
            guard let data else { continue }
            var stale = false
            guard let resolved = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale) else {
                LogFiles.log("VPS4: the folder kept in the \(source) no longer opens")
                continue
            }
            let accessing = resolved.startAccessingSecurityScopedResource()
            guard accessing || FileManager.default.isReadableFile(atPath: resolved.path) else {
                LogFiles.log("VPS4: no access to \(resolved.path) (kept in the \(source))")
                continue
            }
            if stale || source != "settings",
               let fresh = try? resolved.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
                Self.save(fresh)
            }
            opened = resolved
            Self.makeSubfolders(resolved)
            LogFiles.log("VPS4: \(resolved.path) (kept in the \(source))")
            return resolved
        }
        return nil
    }

    /// The folder the player picked. Nil and a reason when it is not one the app can use.
    func choose(_ picked: URL) -> String? {
        guard picked.lastPathComponent.caseInsensitiveCompare(Self.folderName) == .orderedSame else {
            return L("Esa carpeta se llama «\(picked.lastPathComponent)». Elige la carpeta llamada VPS4.",
                     "That folder is called “\(picked.lastPathComponent)”. Choose the folder called VPS4.")
        }
        let accessing = picked.startAccessingSecurityScopedResource()
        guard accessing || FileManager.default.isReadableFile(atPath: picked.path) else {
            return L("No se pudo abrir la carpeta VPS4.", "The VPS4 folder could not be opened.")
        }
        guard let data = try? picked.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) else {
            picked.stopAccessingSecurityScopedResource()
            return L("No se pudo recordar la carpeta VPS4.", "The VPS4 folder could not be remembered.")
        }
        Self.save(data)
        lock.lock()
        opened?.stopAccessingSecurityScopedResource()
        opened = picked
        lock.unlock()
        Self.makeSubfolders(picked)
        LogFiles.log("VPS4: chosen \(picked.path)")
        return nil
    }

    /// Forgets the folder (the files in it stay where they are).
    func forget() {
        lock.lock()
        opened?.stopAccessingSecurityScopedResource()
        opened = nil
        lock.unlock()
        UserDefaults.standard.removeObject(forKey: Self.defaultsKey)
        SecItemDelete(Self.keychainQuery() as CFDictionary)
    }

    /// VPS4/Juegos, where the games' folders go.
    var games: URL? { url?.appendingPathComponent(Self.gamesName, isDirectory: true) }

    /// VPS4/Partidas/<title>, made if it is not there.
    func saves(for title: String) -> URL? {
        guard let folder = url?.appendingPathComponent(Self.savesName, isDirectory: true)
            .appendingPathComponent(title, isDirectory: true) else {
            return nil
        }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    // MARK: - Storage

    private static func makeSubfolders(_ root: URL) {
        // Cachés is AstroVisionPro's (its shader cache); made here too so the folder looks the same.
        for name in [gamesName, savesName, "Cachés"] {
            try? FileManager.default.createDirectory(at: root.appendingPathComponent(name, isDirectory: true),
                                                     withIntermediateDirectories: true)
        }
    }

    private static func save(_ bookmark: Data) {
        UserDefaults.standard.set(bookmark, forKey: defaultsKey)
        SecItemDelete(keychainQuery() as CFDictionary)
        var item = keychainQuery()
        item[kSecValueData as String] = bookmark
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(item as CFDictionary, nil)
    }

    private static func keychainBookmark() -> Data? {
        var query = keychainQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess else {
            return nil
        }
        return result as? Data
    }

    private static func keychainQuery() -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: keychainService,
         kSecAttrAccount as String: keychainAccount]
    }
}
