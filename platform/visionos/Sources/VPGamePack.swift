// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Game packs: a game's x86-64 code translated ahead of time and compiled for the headset
// (tools/scripts/make_game_pack), one file, vpengine.vpgame, in the game's folder in VPS4. The
// app may only run code signed by the same team that signed the app, and SideStore signed the app
// with the player's own development certificate: the app signs the pack with that certificate
// (runtime/vp_codesign.c, the key held by the Security framework) and loads it with dlopen. No JIT,
// no debugger, no extra App ID (the pack is a library of the installed app, not an app).
//
// The certificate comes from SideStore once ("Import from SideStore", its certificate URL scheme,
// as LiveContainer does) or from a .p12 file, and is kept in the keychain. A pack is signed once
// per certificate and kept signed in Library/Application Support/VPEngine/Packs.
//
// The C side the app's bridging header needs: runtime/vp_codesign.h and runtime/vp_emit.h
// (VpPackInfo, vp_runtime_abi), and libVPRuntime linked in.

import Foundation
import Security

enum VPCertificate {
    /// The scheme SideStore calls back with the certificate (the app's Info.plist declares it).
    static var callbackScheme = "vpengine"

    private static let service = "vpengine.signing-certificate"

    struct Summary: Equatable {
        let commonName: String
        let team: String
    }

    enum Problem: Error, LocalizedError {
        case none
        case unreadable(OSStatus)
        case wrongPassword
        case noIdentity
        case keychain(OSStatus)

        var errorDescription: String? {
            switch self {
            case .none:
                return L("No hay certificado: impórtalo desde SideStore.", "There is no certificate: import it from SideStore.")
            case .unreadable(let status):
                return L("El certificado no se pudo leer (\(status)).", "The certificate could not be read (\(status)).")
            case .wrongPassword:
                return L("La contraseña del certificado no es correcta.", "The certificate's password is wrong.")
            case .noIdentity:
                return L("El archivo no contiene un certificado con su clave.", "The file holds no certificate with its key.")
            case .keychain(let status):
                return L("No se pudo guardar el certificado en el llavero (\(status)).", "The certificate could not be kept in the keychain (\(status)).")
            }
        }
    }

    /// Opens SideStore, which hands the certificate back through `callbackScheme`.
    static var sideStoreImportURL: URL? {
        // sidestore://certificate?callback_template=<scheme>://certificate?cert=$(BASE64_CERT)&password=$(PASSWORD)
        let callback = "\(callbackScheme)://certificate?cert=$(BASE64_CERT)&password=$(PASSWORD)"
        guard let encoded = callback.addingPercentEncoding(withAllowedCharacters: .alphanumerics) else { return nil }
        return URL(string: "sidestore://certificate?callback_template=\(encoded)")
    }

    /// The callback from SideStore. Returns nil when `url` is not one, else the result.
    static func handle(url: URL) -> Result<Summary, Error>? {
        guard url.scheme?.lowercased() == callbackScheme.lowercased(), url.host == "certificate" else { return nil }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard let base64 = items.first(where: { $0.name == "cert" })?.value,
              let data = Data(base64Encoded: base64.replacingOccurrences(of: " ", with: "+")) else {
            return .failure(Problem.unreadable(errSecDecode))
        }
        let password = items.first(where: { $0.name == "password" })?.value ?? ""
        return Result { try store(p12: data, password: password) }
    }

    /// A .p12 (PKCS#12) file with the certificate and its key. Checked, then kept.
    @discardableResult
    static func store(p12: Data, password: String) throws -> Summary {
        let identity = try identity(p12: p12, password: password)
        let summary = try describe(identity)
        try keep(account: "p12", data: p12)
        try keep(account: "password", data: Data(password.utf8))
        return summary
    }

    static func remove() {
        for account in ["p12", "password"] {
            let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                        kSecAttrService as String: service, kSecAttrAccount as String: account]
            SecItemDelete(query as CFDictionary)
        }
    }

    /// The certificate kept, if any.
    static var current: Summary? {
        guard let identity = try? storedIdentity() else { return nil }
        return try? describe(identity)
    }

    static func storedIdentity() throws -> SecIdentity {
        guard let p12 = read(account: "p12") else { throw Problem.none }
        let password = read(account: "password").flatMap { String(data: $0, encoding: .utf8) } ?? ""
        return try identity(p12: p12, password: password)
    }

    /// Also the certificates the .p12 carried besides the leaf (an intermediate, sometimes).
    static func storedChain() -> [SecCertificate] {
        guard let p12 = read(account: "p12") else { return [] }
        let password = read(account: "password").flatMap { String(data: $0, encoding: .utf8) } ?? ""
        var items: CFArray?
        guard SecPKCS12Import(p12 as CFData, [kSecImportExportPassphrase as String: password] as CFDictionary, &items) == errSecSuccess,
              let first = (items as? [[String: Any]])?.first,
              let chain = first[kSecImportItemCertChain as String] as? [SecCertificate] else { return [] }
        return Array(chain.dropFirst())
    }

    private static func identity(p12: Data, password: String) throws -> SecIdentity {
        var items: CFArray?
        let status = SecPKCS12Import(p12 as CFData, [kSecImportExportPassphrase as String: password] as CFDictionary, &items)
        if status == errSecAuthFailed { throw Problem.wrongPassword }
        guard status == errSecSuccess else { throw Problem.unreadable(status) }
        guard let first = (items as? [[String: Any]])?.first,
              let value = first[kSecImportItemIdentity as String] else { throw Problem.noIdentity }
        return value as! SecIdentity // swiftlint:disable:this force_cast (a CF type: the cast cannot fail)
    }

    private static func describe(_ identity: SecIdentity) throws -> Summary {
        var certificate: SecCertificate?
        guard SecIdentityCopyCertificate(identity, &certificate) == errSecSuccess, let certificate else { throw Problem.noIdentity }
        let name = SecCertificateCopySubjectSummary(certificate) as String? ?? "?"
        let der = SecCertificateCopyData(certificate) as Data
        let team = der.withUnsafeBytes { raw -> String in
            var out = [CChar](repeating: 0, count: 64)
            return vp_codesign_cert_team(raw.bindMemory(to: UInt8.self).baseAddress, der.count, &out, out.count) == 0
                ? String(cString: out) : "?"
        }
        return Summary(commonName: name, team: team)
    }

    private static func keep(account: String, data: Data) throws {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service, kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
        var add = query
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(add as CFDictionary, nil)
        guard status == errSecSuccess else { throw Problem.keychain(status) }
    }

    private static func read(account: String) -> Data? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else { return nil }
        return item as? Data
    }
}

enum VPGamePack {
    static let fileName = "vpengine.vpgame"
    /// Where the app's messages go (the app sets its own log).
    static var log: (String) -> Void = { NSLog("%@", $0) }

    struct Loaded: Equatable {
        let title: String
        let modules: Int
        let path: String
    }

    enum Problem: Error, LocalizedError {
        case noPack
        case certificate(Error)
        case wrongTeam(app: String, certificate: String)
        case signing(String)
        case loading(String)
        case notAPack
        case wrongVersion(pack: UInt32, app: Int32)

        var errorDescription: String? {
            switch self {
            case .noPack:
                return L("No hay \(VPGamePack.fileName) en la carpeta del juego: créalo en el PC con make_game_pack y cópialo junto a eboot.bin.",
                         "There is no \(VPGamePack.fileName) in the game's folder: make it on the PC with make_game_pack and copy it next to eboot.bin.")
            case .certificate(let error):
                return error.localizedDescription
            case .wrongTeam(let app, let certificate):
                return L("El certificado (equipo \(certificate)) no es con el que SideStore firmó la app (equipo \(app)): impórtalo otra vez desde SideStore.",
                         "The certificate (team \(certificate)) is not the one SideStore signed the app with (team \(app)): import it again from SideStore.")
            case .signing(let message):
                return L("No se pudo firmar el juego: \(message)", "The game could not be signed: \(message)")
            case .loading(let message):
                return L("El sistema no cargó el juego: \(message)", "The system did not load the game: \(message)")
            case .notAPack:
                return L("\(VPGamePack.fileName) no es un paquete de VPEngine.", "\(VPGamePack.fileName) is not a VPEngine pack.")
            case .wrongVersion(let pack, let app):
                return L("El paquete es de otra versión del motor (\(pack), la app usa \(app)): vuelve a crearlo con las herramientas de esta versión.",
                         "The pack is from another engine version (\(pack), the app uses \(app)): make it again with this version's tools.")
            }
        }
    }

    private static var loaded: [String: Loaded] = [:]

    /// A pack already loaded in this process (a library cannot be unloaded and loaded again
    /// changed: the app must be restarted to use a new one).
    static func alreadyLoaded() -> Loaded? {
        lock.lock()
        defer { lock.unlock() }
        return loaded.values.first
    }
    private static let lock = NSLock()

    static func packURL(in gameFolder: URL) -> URL? {
        let url = gameFolder.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Signs the game's pack if needed and loads it: its modules are then registered with the
    /// runtime and the engine attaches them when the emulator loads the game's files. Slow the
    /// first time (copy and signature of a few hundred MB): call it off the main thread.
    static func load(gameFolder: URL, progress: ((String) -> Void)? = nil) throws -> Loaded {
        guard let source = packURL(in: gameFolder) else { throw Problem.noPack }
        return try load(pack: source, progress: progress)
    }

    /// The same for a pack anywhere (one converted on the headset, VPConversion.swift).
    static func load(pack source: URL, progress: ((String) -> Void)? = nil) throws -> Loaded {
        guard FileManager.default.fileExists(atPath: source.path) else { throw Problem.noPack }
        lock.lock()
        defer { lock.unlock() }
        if let done = loaded[source.path] { return done }

        let identity: SecIdentity
        do { identity = try VPCertificate.storedIdentity() } catch { throw Problem.certificate(error) }
        var certificate: SecCertificate?
        var key: SecKey?
        guard SecIdentityCopyCertificate(identity, &certificate) == errSecSuccess, let certificate,
              SecIdentityCopyPrivateKey(identity, &key) == errSecSuccess, let key else {
            throw Problem.certificate(VPCertificate.Problem.noIdentity)
        }
        let leaf = SecCertificateCopyData(certificate) as Data

        // The signed copy: one per pack and certificate, made again when either changes.
        let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        var digest = [UInt8](repeating: 0, count: 32)
        let identityText = "\(source.path)|\(size)|\(modified)|\(leaf.base64EncodedString())"
        identityText.withCString { vp_sha256($0, strlen($0), &digest) }
        let tag = digest.prefix(12).map { String(format: "%02x", $0) }.joined()
        let folder = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VPEngine/Packs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let signed = folder.appendingPathComponent("\(tag).dylib")

        if !FileManager.default.fileExists(atPath: signed.path) {
            // Only this pack's copies: the old ones (other versions, other certificates) go.
            for old in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] {
                try? FileManager.default.removeItem(at: old)
            }
            progress?(L("Copiando el juego…", "Copying the game…"))
            let partial = folder.appendingPathComponent("\(tag).partial")
            try? FileManager.default.removeItem(at: partial)
            try FileManager.default.copyItem(at: source, to: partial)
            progress?(L("Firmando el juego con tu certificado…", "Signing the game with your certificate…"))
            try sign(path: partial.path, leaf: leaf, certificate: certificate, key: key)
            try FileManager.default.moveItem(at: partial, to: signed)
            log("VPEngine: signed \(source.lastPathComponent) (\(size >> 20) MB) as \(signed.lastPathComponent)")
        }

        progress?(L("Cargando el juego…", "Loading the game…"))
        guard let handle = dlopen(signed.path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "?"
            log("VPEngine: dlopen failed: \(reason)")
            // A copy the system refuses is no use next time either.
            try? FileManager.default.removeItem(at: signed)
            throw Problem.loading(reason)
        }
        guard let symbol = dlsym(handle, "vp_pack_info") else { throw Problem.notAPack }
        let info = symbol.assumingMemoryBound(to: VpPackInfo.self).pointee
        guard info.magic == VP_PACK_MAGIC else { throw Problem.notAPack }
        guard Int32(info.abi) == vp_runtime_abi() else { throw Problem.wrongVersion(pack: info.abi, app: vp_runtime_abi()) }
        let result = Loaded(title: info.title.map { String(cString: $0) } ?? "?", modules: Int(info.module_count), path: signed.path)
        log("VPEngine: pack \(result.title) loaded, \(result.modules) translated modules")
        loaded[source.path] = result
        return result
    }

    private static func sign(path: String, leaf: Data, certificate: SecCertificate, key: SecKey) throws {
        // The app's own team: the pack must carry the same.
        if let executable = Bundle.main.executablePath {
            var appTeam = [CChar](repeating: 0, count: 64)
            if vp_codesign_file_team(executable, &appTeam, appTeam.count) == 0 {
                var certificateTeam = [CChar](repeating: 0, count: 64)
                let found = leaf.withUnsafeBytes {
                    vp_codesign_cert_team($0.bindMemory(to: UInt8.self).baseAddress, leaf.count, &certificateTeam, certificateTeam.count)
                }
                let app = String(cString: appTeam), mine = String(cString: certificateTeam)
                if found == 0 && app != mine {
                    throw Problem.wrongTeam(app: app, certificate: mine)
                }
            }
        }
        let chain = Self.chain(for: certificate).map { SecCertificateCopyData($0) as Data }
        var error = [CChar](repeating: 0, count: 256)
        let chainBuffers = chain.map { [UInt8]($0) }
        let session: OpaquePointer? = leaf.withUnsafeBytes { leafBytes in
            withArrayOfPointers(chainBuffers) { pointers, lengths in
                vp_codesign_begin(path, "com.vpengine.pack", leafBytes.bindMemory(to: UInt8.self).baseAddress, leaf.count,
                                  pointers, lengths, Int32(chainBuffers.count), &error, error.count)
            }
        }
        guard let session else { throw Problem.signing(String(cString: error)) }
        var length = 0
        guard let toSign = vp_codesign_to_sign(session, &length) else {
            vp_codesign_abort(session)
            throw Problem.signing("no attributes")
        }
        let message = Data(bytes: toSign, count: length)
        let isEC = (SecKeyCopyAttributes(key) as? [String: Any])?[kSecAttrKeyType as String] as? String == (kSecAttrKeyTypeECSECPrimeRandom as String)
        let algorithm: SecKeyAlgorithm = isEC ? .ecdsaSignatureMessageX962SHA256 : .rsaSignatureMessagePKCS1v15SHA256
        var cfError: Unmanaged<CFError>?
        guard let signature = SecKeyCreateSignature(key, algorithm, message as CFData, &cfError) as Data? else {
            vp_codesign_abort(session)
            throw Problem.signing(cfError?.takeRetainedValue().localizedDescription ?? "SecKeyCreateSignature")
        }
        let status = signature.withUnsafeBytes {
            vp_codesign_finish(session, $0.bindMemory(to: UInt8.self).baseAddress, signature.count, isEC ? 1 : 0, &error, error.count)
        }
        guard status == 0 else { throw Problem.signing(String(cString: error)) }
    }

    /// The certificates between the leaf and Apple's root: what the .p12 carried, else Apple's
    /// intermediates and root in the app's bundle (VPEngineCertificates/*.cer), picked by issuer.
    private static func chain(for leaf: SecCertificate) -> [SecCertificate] {
        var candidates = VPCertificate.storedChain()
        let bundled = Bundle.main.urls(forResourcesWithExtension: "cer", subdirectory: "VPEngineCertificates") ?? []
        for url in bundled {
            if let data = try? Data(contentsOf: url), let c = SecCertificateCreateWithData(nil, data as CFData) {
                candidates.append(c)
            }
        }
        var chain: [SecCertificate] = []
        var current = leaf
        for _ in 0..<4 {
            guard let issuer = SecCertificateCopyNormalizedIssuerSequence(current) as Data?,
                  let next = candidates.first(where: { (SecCertificateCopyNormalizedSubjectSequence($0) as Data?) == issuer }),
                  !chain.contains(where: { CFEqual($0, next) }), !CFEqual(next, current) else { break }
            chain.append(next)
            if (SecCertificateCopyNormalizedSubjectSequence(next) as Data?) == (SecCertificateCopyNormalizedIssuerSequence(next) as Data?) { break } // the root
            current = next
        }
        log("VPEngine: signing with \(chain.count) certificates after the leaf (\(bundled.count) bundled)")
        return chain
    }
}

/// C arrays of pointers and lengths for a list of byte buffers, valid inside `body`.
private func withArrayOfPointers<R>(_ buffers: [[UInt8]], _ body: (UnsafePointer<UnsafePointer<UInt8>?>?, UnsafePointer<Int>?) -> R) -> R {
    var pointers: [UnsafePointer<UInt8>?] = []
    var lengths: [Int] = []
    var copies: [UnsafeMutablePointer<UInt8>] = []
    for b in buffers {
        let p = UnsafeMutablePointer<UInt8>.allocate(capacity: max(b.count, 1))
        p.initialize(from: b, count: b.count)
        copies.append(p)
        pointers.append(UnsafePointer(p))
        lengths.append(b.count)
    }
    defer { copies.forEach { $0.deallocate() } }
    return pointers.withUnsafeBufferPointer { pp in
        lengths.withUnsafeBufferPointer { lp in body(pp.baseAddress, lp.baseAddress) }
    }
}
