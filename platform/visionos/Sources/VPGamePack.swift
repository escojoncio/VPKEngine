// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Game packs: a game's x86-64 code translated ahead of time and compiled for the headset
// (tools/scripts/make_game_pack), one file, vpkengine.vpgame, in the game's folder in VPS4. The
// app may only run code signed by the same team that signed the app, and SideStore signed the app
// with the player's own development certificate: the app signs the pack with that certificate
// (runtime/vp_codesign.c, the key held by the Security framework) and loads it with dlopen. No JIT,
// no debugger, no extra App ID (the pack is a library of the installed app, not an app).
//
// The certificate comes from SideStore once ("Import from SideStore", its certificate URL scheme,
// as LiveContainer does) or from a .p12 file, and is kept in the keychain. A pack is signed once
// per certificate and kept signed in Library/Application Support/VPKEngine/Packs.
//
// The C side the app's bridging header needs: runtime/vp_codesign.h and runtime/vp_emit.h
// (VpPackInfo, vp_runtime_abi), and libVPRuntime linked in.

import Foundation
import Security

enum VPCertificate {
    /// The scheme SideStore calls back with the certificate (the app's Info.plist declares it).
    static var callbackScheme = "vpkengine"

    private static let service = "vpkengine.signing-certificate"

    struct Summary: Equatable {
        let commonName: String
        let team: String
    }

    /// The certificate and key a pack is signed with (from Apple's .p12 reader or ours).
    struct SigningIdentity {
        let certificate: SecCertificate
        let key: SecKey
        /// The other certificates the .p12 carried (an intermediate, sometimes).
        let chain: [SecCertificate]
    }

    /// Where the certificate's messages go (the app sets its own log).
    static var log: (String) -> Void = { NSLog("%@", $0) }

    enum Problem: Error, LocalizedError {
        case missing
        case unreadable(OSStatus, String)
        case wrongPassword
        case noIdentity
        case keychain(OSStatus)

        var errorDescription: String? {
            switch self {
            case .missing:
                return L("No hay certificado: impórtalo desde SideStore.", "There is no certificate: import it from SideStore.")
            case .unreadable(let status, let detail):
                return L("El certificado no se pudo leer (\(status)): \(detail).", "The certificate could not be read (\(status)): \(detail).")
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
            return .failure(Problem.unreadable(errSecDecode, "SideStore's answer has no base64 certificate"))
        }
        let password = items.first(where: { $0.name == "password" })?.value ?? ""
        return Result { try store(p12: data, password: password) }
    }

    /// A .p12 (PKCS#12) file with the certificate and its key. Checked, then kept.
    @discardableResult
    static func store(p12: Data, password: String) throws -> Summary {
        let identity = try signingIdentity(p12: p12, password: password)
        let summary = describe(identity.certificate)
        // Whether the app's profile allows this certificate (what a pack signed with it needs).
        let leafData = SecCertificateCopyData(identity.certificate) as Data
        SigningDiagnosis.profile(leaf: leafData).split(separator: "\n").forEach { log(String($0)) }
        try keep(account: "p12", data: p12)
        try keep(account: "password", data: Data(password.utf8))
        return summary
    }

    static func remove() {
        // Also under the old service: otherwise read(account:) would bring it back.
        for svc in [service, VPKLegacy.keychainService] {
            for account in ["p12", "password"] {
                let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                            kSecAttrService as String: svc, kSecAttrAccount as String: account]
                SecItemDelete(query as CFDictionary)
            }
        }
    }

    /// The certificate kept, if any.
    static var current: Summary? {
        guard let identity = try? storedIdentity() else { return nil }
        return describe(identity.certificate)
    }

    static func storedIdentity() throws -> SigningIdentity {
        guard let p12 = read(account: "p12") else { throw Problem.missing }
        let password = read(account: "password").flatMap { String(data: $0, encoding: .utf8) } ?? ""
        return try signingIdentity(p12: p12, password: password, quiet: true)
    }

    /// Apple's reader first; ours (runtime/vp_pkcs12.c) for what it does not read: the AES
    /// encryption (PBES2) of OpenSSL 3 and the tools built on it, iloader's among them.
    private static func signingIdentity(p12: Data, password: String, quiet: Bool = false) throws -> SigningIdentity {
        var items: CFArray?
        let status = SecPKCS12Import(p12 as CFData, [kSecImportExportPassphrase as String: password] as CFDictionary, &items)
        if status == errSecSuccess, let first = (items as? [[String: Any]])?.first,
           let value = first[kSecImportItemIdentity as String] {
            let identity = value as! SecIdentity // swiftlint:disable:this force_cast (a CF type: the cast cannot fail)
            var certificate: SecCertificate?
            var key: SecKey?
            if SecIdentityCopyCertificate(identity, &certificate) == errSecSuccess, let certificate,
               SecIdentityCopyPrivateKey(identity, &key) == errSecSuccess, let key {
                let chain = (first[kSecImportItemCertChain as String] as? [SecCertificate]) ?? []
                if !quiet { log("VPKEngine: certificate read by the system's PKCS#12 reader") }
                return SigningIdentity(certificate: certificate, key: key, chain: Array(chain.dropFirst()))
            }
        }
        if !quiet { log("VPKEngine: the system's PKCS#12 reader gave \(status); reading it with VPKEngine's own") }

        var parsed = VpPkcs12()
        var error = [CChar](repeating: 0, count: 256)
        let rc = p12.withUnsafeBytes { raw in
            vp_pkcs12_read(raw.bindMemory(to: UInt8.self).baseAddress, p12.count, password, &parsed, &error, error.count)
        }
        defer { vp_pkcs12_free(&parsed) }
        let detail = String(cString: error)
        if rc != 0 && !quiet { log("VPKEngine: VPKEngine's PKCS#12 reader: \(rc), \(detail)") }
        // Apple's reader decides a wrong password when ours cannot read the file at all.
        if status == errSecAuthFailed && rc != 0 && rc != -4 { throw Problem.wrongPassword }
        switch rc {
        case 0: break
        case -2: throw Problem.wrongPassword
        case -4: throw Problem.noIdentity
        default: throw Problem.unreadable(status, detail)
        }
        guard let keyBytes = parsed.key else { throw Problem.noIdentity }
        let keyData = Data(bytes: keyBytes, count: parsed.key_len)
        let ec = parsed.key_type == 2
        let attributes: [String: Any] = [
            kSecAttrKeyType as String: ec ? kSecAttrKeyTypeECSECPrimeRandom : kSecAttrKeyTypeRSA,
            kSecAttrKeyClass as String: kSecAttrKeyClassPrivate,
        ]
        var cfError: Unmanaged<CFError>?
        guard let key = SecKeyCreateWithData(keyData as CFData, attributes as CFDictionary, &cfError) else {
            let reason = cfError?.takeRetainedValue().localizedDescription ?? "SecKeyCreateWithData"
            log("VPKEngine: the key could not be made: \(reason)")
            throw Problem.unreadable(status, reason)
        }
        var certificates: [SecCertificate] = []
        for i in 0..<Int(parsed.cert_count) {
            var length = 0
            guard let bytes = vp_pkcs12_cert(&parsed, Int32(i), &length),
                  let c = SecCertificateCreateWithData(nil, Data(bytes: bytes, count: length) as CFData) else { continue }
            certificates.append(c)
        }
        // The key's certificate: the one whose public key is the key's (the file may list a CA first
        // and carry no localKeyID to say which).
        let publicKey = SecKeyCopyPublicKey(key).flatMap { SecKeyCopyExternalRepresentation($0, nil) as Data? }
        let leafIndex = certificates.firstIndex { certificate in
            guard let publicKey, let theirs = SecCertificateCopyKey(certificate) else { return false }
            return (SecKeyCopyExternalRepresentation(theirs, nil) as Data?) == publicKey
        }
        guard let leafIndex else {
            if !quiet { log("VPKEngine: no certificate in the file matches its private key") }
            throw Problem.noIdentity
        }
        let leaf = certificates.remove(at: leafIndex)
        if !quiet { log("VPKEngine: certificate read by VPKEngine's PKCS#12 reader (\(ec ? "EC" : "RSA") key, \(certificates.count + 1) certificates)") }
        return SigningIdentity(certificate: leaf, key: key, chain: certificates)
    }

    private static func describe(_ certificate: SecCertificate) -> Summary {
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
        if let data = read(account: account, service: service) { return data }
        // Kept by a build from before the rename: moved to the current service.
        guard let data = read(account: account, service: VPKLegacy.keychainService) else { return nil }
        if (try? keep(account: account, data: data)) != nil {
            let old: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                      kSecAttrService as String: VPKLegacy.keychainService,
                                      kSecAttrAccount as String: account]
            SecItemDelete(old as CFDictionary)
        }
        return data
    }

    private static func read(account: String, service: String) -> Data? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else { return nil }
        return item as? Data
    }
}

/// The engine was called VPEngine before VPKEngine. What an earlier build left on the headset
/// under the old name is taken over the first time it is needed, so renaming costs the player
/// nothing: the conversion folders (VPS4/VPEngine, Application Support/VPEngine), saved settings,
/// the signing certificate in the keychain and game packs named vpengine.vpgame.
enum VPKLegacy {
    static let folderName = "VPEngine"
    static let packFileName = "vpengine.vpgame"
    static let keychainService = "vpengine.signing-certificate"

    /// `parent/VPKEngine`; an existing `parent/VPEngine` is renamed to it first (same volume: no
    /// copy). If that fails the old folder keeps being used, so nothing is ever lost.
    private static let folderLock = NSLock()   // several threads ask; only one may rename

    static func folder(in parent: URL) -> URL {
        folderLock.lock()
        defer { folderLock.unlock() }
        let new = parent.appendingPathComponent("VPKEngine", isDirectory: true)
        let old = parent.appendingPathComponent(folderName, isDirectory: true)
        let manager = FileManager.default
        guard !manager.fileExists(atPath: new.path), manager.fileExists(atPath: old.path) else { return new }
        do {
            try manager.moveItem(at: old, to: new)
            NSLog("%@", "[VPKEngine] \(old.path) renamed to \(new.lastPathComponent)")
            return new
        } catch {
            NSLog("%@", "[VPKEngine] could not rename \(old.path) (\(error.localizedDescription)): still used as is")
            return old
        }
    }

    /// A setting saved under its old key moves to the new one.
    static func takeOverDefault(_ key: String, from old: String) {
        let defaults = UserDefaults.standard
        guard let value = defaults.object(forKey: old) else { return }
        if defaults.object(forKey: key) == nil { defaults.set(value, forKey: key) }
        defaults.removeObject(forKey: old)
    }
}

enum VPGamePack {
    static let fileName = "vpkengine.vpgame"
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
                return L("\(VPGamePack.fileName) no es un paquete de VPKEngine.", "\(VPGamePack.fileName) is not a VPKEngine pack.")
            case .wrongVersion(let pack, let app):
                return L("El paquete es de otra versión del motor (\(pack), la app usa \(app)): vuelve a crearlo con las herramientas de esta versión.",
                         "The pack is from another engine version (\(pack), the app uses \(app)): make it again with this version's tools.")
            }
        }
    }

    private static var loaded: [String: Loaded] = [:]
    /// Packs (by copy tag) whose every signature variant the system refused in this run: not
    /// signed and tried again on every tap.
    private static var failed: [String: String] = [:]

    /// The ways of signing a pack, tried in order until the system accepts one (the one accepted
    /// is remembered and tried first next time). All carry the app's identifier; they differ in
    /// whether the app's own requirement set is carried and in the page size hashed.
    struct SignatureVariant {
        let name: String
        let identifier: String
        let requirements: Data?
        let pageShift: Int32

        var summary: String {
            "identifier \(identifier), \(requirements.map { "the app's requirement set (\($0.count) bytes)" } ?? "an empty requirement set"), \(1 << (pageShift - 10)) KiB pages"
        }

        private static let defaultsKey = "VPKEngine.signatureVariant"

        static func ordered() -> [SignatureVariant] {
            let identifier = Bundle.main.bundleIdentifier ?? "com.vpkengine.pack"
            var requirements: Data?
            if let executable = Bundle.main.executablePath {
                var buffer = [UInt8](repeating: 0, count: 4096)
                let n = vp_codesign_file_requirements(executable, &buffer, buffer.count)
                if n >= 12 { requirements = Data(buffer.prefix(Int(n))) }
            }
            var all: [SignatureVariant] = []
            if let requirements {
                all.append(SignatureVariant(name: "req4k", identifier: identifier, requirements: requirements, pageShift: 12))
                all.append(SignatureVariant(name: "req16k", identifier: identifier, requirements: requirements, pageShift: 14))
            }
            all.append(SignatureVariant(name: "plain4k", identifier: identifier, requirements: nil, pageShift: 12))
            all.append(SignatureVariant(name: "plain16k", identifier: identifier, requirements: nil, pageShift: 14))
            VPKLegacy.takeOverDefault(defaultsKey, from: "VPEngine.signatureVariant")
            if let remembered = UserDefaults.standard.string(forKey: defaultsKey),
               let index = all.firstIndex(where: { $0.name == remembered }) {
                all.insert(all.remove(at: index), at: 0)
            }
            return all
        }

        static func remember(_ variant: SignatureVariant) {
            UserDefaults.standard.set(variant.name, forKey: defaultsKey)
        }
    }

    /// A pack already loaded in this process (a library cannot be unloaded and loaded again
    /// changed: the app must be restarted to use a new one).
    static func alreadyLoaded() -> Loaded? {
        lock.lock()
        defer { lock.unlock() }
        return loaded.values.first
    }
    private static let lock = NSLock()

    static func packURL(in gameFolder: URL) -> URL? {
        for name in [fileName, VPKLegacy.packFileName] {
            let url = gameFolder.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
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

        let identity: VPCertificate.SigningIdentity
        do { identity = try VPCertificate.storedIdentity() } catch { throw Problem.certificate(error) }
        let certificate = identity.certificate, key = identity.key
        let leaf = SecCertificateCopyData(certificate) as Data

        // The signed copy: one per pack, certificate and signature variant, made again when any
        // changes.
        let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        var digest = [UInt8](repeating: 0, count: 32)
        let identityText = "\(source.path)|\(size)|\(modified)|\(leaf.base64EncodedString())"
        identityText.withCString { vp_sha256($0, strlen($0), &digest) }
        let tag = digest.prefix(12).map { String(format: "%02x", $0) }.joined()
        if let reason = failed[tag] { throw Problem.loading(reason) } // every variant was refused in this run already
        let folder = VPKLegacy.folder(in: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0])
            .appendingPathComponent("Packs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        // Only this pack's copies stay: the old ones (other versions, other certificates) go.
        for old in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        where !old.lastPathComponent.hasPrefix(tag) {
            try? FileManager.default.removeItem(at: old)
        }

        var handle: UnsafeMutableRawPointer?
        var lastReason = "?"
        var lastSigned: URL?
        var loadedPath = ""
        for variant in SignatureVariant.ordered() {
            let signed = folder.appendingPathComponent("\(tag)-\(variant.name).dylib")
            if !FileManager.default.fileExists(atPath: signed.path) {
                progress?(L("Copiando el juego…", "Copying the game…"))
                let partial = folder.appendingPathComponent("\(tag)-\(variant.name).partial")
                try? FileManager.default.removeItem(at: partial)
                try FileManager.default.copyItem(at: source, to: partial)
                progress?(L("Firmando el juego con tu certificado…", "Signing the game with your certificate…"))
                do {
                    try sign(path: partial.path, variant: variant, leaf: leaf, certificate: certificate, key: key, carried: identity.chain)
                } catch {
                    try? FileManager.default.removeItem(at: partial)
                    if let previous = lastSigned { try? FileManager.default.removeItem(at: previous) }
                    throw error
                }
                try FileManager.default.moveItem(at: partial, to: signed)
                log("VPKEngine: signed \(source.lastPathComponent) (\(size >> 20) MB) as \(signed.lastPathComponent): \(variant.summary)")
            }
            progress?(L("Cargando el juego…", "Loading the game…"))
            if let h = dlopen(signed.path, RTLD_NOW | RTLD_LOCAL) {
                handle = h
                loadedPath = signed.path
                log("VPKEngine: the system accepted the signature \(variant.name)")
                SignatureVariant.remember(variant)
                if let previous = lastSigned { try? FileManager.default.removeItem(at: previous) }
                lastSigned = nil
                break
            }
            lastReason = dlerror().map { String(cString: $0) } ?? "?"
            log("VPKEngine: dlopen refused the signature \(variant.name): \(lastReason)")
            if let previous = lastSigned { try? FileManager.default.removeItem(at: previous) }
            lastSigned = signed // kept until the next variant is tried, for the diagnosis
            // Another signature only helps when the signature was what the system refused.
            if !lastReason.contains("code signature") { break }
        }
        guard let handle else {
            if let signed = lastSigned {
                SigningDiagnosis.report(pack: signed, leaf: leaf).split(separator: "\n").forEach { log(String($0)) }
                // A copy the system refuses is no use next time either.
                try? FileManager.default.removeItem(at: signed)
            }
            failed[tag] = lastReason
            throw Problem.loading(lastReason)
        }
        guard let symbol = dlsym(handle, "vp_pack_info") else { throw Problem.notAPack }
        let info = symbol.assumingMemoryBound(to: VpPackInfo.self).pointee
        guard info.magic == VP_PACK_MAGIC else { throw Problem.notAPack }
        guard Int32(info.abi) == vp_runtime_abi() else { throw Problem.wrongVersion(pack: info.abi, app: vp_runtime_abi()) }
        let result = Loaded(title: info.title.map { String(cString: $0) } ?? "?", modules: Int(info.module_count), path: loadedPath)
        log("VPKEngine: pack \(result.title) loaded, \(result.modules) translated modules")
        loaded[source.path] = result
        return result
    }

    // MARK: - Translations on demand

    private static let fragmentLock = NSLock()
    private static var fragmentsLoaded: Set<String> = []

    /// A library translated on demand while the game runs (vpconvert's vp_convert_fragment, in the
    /// game's VPS4/VPKEngine/<game>/ondemand): signed like the pack (the variant the system accepted
    /// first) into the app's own folder and loaded; its module registers itself and the engine
    /// attaches it. Quick (a small library).
    static func loadFragment(_ source: URL) throws {
        fragmentLock.lock()
        defer { fragmentLock.unlock() }
        if fragmentsLoaded.contains(source.path) { return }
        let identity: VPCertificate.SigningIdentity
        do { identity = try VPCertificate.storedIdentity() } catch { throw Problem.certificate(error) }
        let leaf = SecCertificateCopyData(identity.certificate) as Data
        let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        var digest = [UInt8](repeating: 0, count: 32)
        let identityText = "\(source.path)|\(size)|\(modified)|\(leaf.base64EncodedString())"
        identityText.withCString { vp_sha256($0, strlen($0), &digest) }
        let tag = digest.prefix(12).map { String(format: "%02x", $0) }.joined()
        let folder = VPKLegacy.folder(in: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0])
            .appendingPathComponent("OnDemand", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        // This library's copies made for an earlier version of it (another size, date or certificate)
        // stay otherwise for good: "<name>-<24 hex of the tag>-<variant>.dylib" with another tag goes.
        let stem = source.deletingPathExtension().lastPathComponent
        for old in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] {
            let file = old.lastPathComponent
            guard file.hasPrefix(stem + "-") else { continue }
            let rest = file.dropFirst(stem.count + 1)
            let isTag = rest.count > 25 && rest.prefix(24).allSatisfy { $0.isHexDigit } && rest[rest.index(rest.startIndex, offsetBy: 24)] == "-"
            if isTag && !rest.hasPrefix(tag + "-") {
                try? FileManager.default.removeItem(at: old)
            }
        }
        var lastReason = "?"
        for variant in SignatureVariant.ordered() {
            let signed = folder.appendingPathComponent("\(stem)-\(tag)-\(variant.name).dylib")
            if !FileManager.default.fileExists(atPath: signed.path) {
                let partial = signed.appendingPathExtension("partial")
                try? FileManager.default.removeItem(at: partial)
                try FileManager.default.copyItem(at: source, to: partial)
                do {
                    try sign(path: partial.path, variant: variant, leaf: leaf, certificate: identity.certificate, key: identity.key, carried: identity.chain)
                } catch {
                    try? FileManager.default.removeItem(at: partial)
                    throw error
                }
                try FileManager.default.moveItem(at: partial, to: signed)
            }
            if let handle = dlopen(signed.path, RTLD_NOW | RTLD_LOCAL) {
                SignatureVariant.remember(variant)
                guard let symbol = dlsym(handle, "vp_pack_info") else { throw Problem.notAPack }
                let info = symbol.assumingMemoryBound(to: VpPackInfo.self).pointee
                guard info.magic == VP_PACK_MAGIC else { throw Problem.notAPack }
                guard Int32(info.abi) == vp_runtime_abi() else { throw Problem.wrongVersion(pack: info.abi, app: vp_runtime_abi()) }
                fragmentsLoaded.insert(source.path)
                log("VPKEngine: loaded \(source.lastPathComponent) (translated on demand)")
                return
            }
            lastReason = dlerror().map { String(cString: $0) } ?? "?"
            try? FileManager.default.removeItem(at: signed)
            if !lastReason.contains("code signature") { break }
        }
        throw Problem.loading(lastReason)
    }

    /// Every library translated on demand before for this game (a folder of them), loaded now so
    /// that the game does not stop to translate them again. Ones that fail are logged and skipped.
    static func loadFragments(in folder: URL) -> Int {
        let files = ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "dylib" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var loaded = 0
        for file in files {
            do {
                try loadFragment(file)
                loaded += 1
            } catch {
                log("VPKEngine: \(file.lastPathComponent) not loaded: \(error.localizedDescription)")
            }
        }
        return loaded
    }

    private static func sign(path: String, variant: SignatureVariant, leaf: Data, certificate: SecCertificate, key: SecKey, carried: [SecCertificate]) throws {
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
        let chain = Self.chain(for: certificate, carried: carried).map { SecCertificateCopyData($0) as Data }
        var error = [CChar](repeating: 0, count: 256)
        let chainBuffers = chain.map { [UInt8]($0) }
        let requirements = [UInt8](variant.requirements ?? Data())
        let session: OpaquePointer? = leaf.withUnsafeBytes { leafBytes in
            withArrayOfPointers(chainBuffers) { pointers, lengths in
                requirements.withUnsafeBufferPointer { req in
                    vp_codesign_begin_ex(path, variant.identifier, requirements.isEmpty ? nil : req.baseAddress, requirements.count,
                                         variant.pageShift, leafBytes.bindMemory(to: UInt8.self).baseAddress, leaf.count,
                                         pointers, lengths, Int32(chainBuffers.count), &error, error.count)
                }
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
    /// intermediates and root in the app's bundle (VPKEngineCertificates/*.cer), picked by issuer.
    private static func chain(for leaf: SecCertificate, carried: [SecCertificate]) -> [SecCertificate] {
        var candidates = carried
        let bundled = Bundle.main.urls(forResourcesWithExtension: "cer", subdirectory: "VPKEngineCertificates") ?? []
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
        log("VPKEngine: signing with \(chain.count) certificates after the leaf (\(bundled.count) bundled)")
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

/// Why the system may refuse a pack's signature, without a Mac: the app's own signature (which the
/// system accepts) next to the pack's, and whether the certificate is one the app's provisioning
/// profile allows (the system checks a developer-signed library against it).
enum SigningDiagnosis {
    static func certificateLine(_ der: Data) -> String {
        var out = [CChar](repeating: 0, count: 256)
        der.withUnsafeBytes { vp_codesign_cert_describe($0.bindMemory(to: UInt8.self).baseAddress, der.count, &out, out.count) }
        return String(cString: out)
    }

    static func describe(_ path: String) -> String {
        var out = [CChar](repeating: 0, count: 1 << 16)
        vp_codesign_describe(path, &out, out.count)
        return String(cString: out)
    }

    /// The app's embedded.mobileprovision: its name, team, expiry, devices and certificates, and
    /// whether `leaf` is among them.
    static func profile(leaf: Data?) -> String {
        guard let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
              let data = try? Data(contentsOf: url) else { return "provisioning profile: none in the app" }
        guard let start = data.range(of: Data("<?xml".utf8)),
              let end = data.range(of: Data("</plist>".utf8), in: start.lowerBound..<data.endIndex),
              let dict = (try? PropertyListSerialization.propertyList(from: data[start.lowerBound..<end.upperBound], format: nil)) as? [String: Any]
        else { return "provisioning profile: \(data.count) bytes, its property list could not be read" }
        var lines: [String] = []
        let teams = (dict["TeamIdentifier"] as? [String])?.joined(separator: ",") ?? "?"
        let devices = (dict["ProvisionedDevices"] as? [Any])?.count ?? 0
        let expiry = (dict["ExpirationDate"] as? Date).map { "\($0)" } ?? "?"
        lines.append("provisioning profile \"\(dict["Name"] as? String ?? "?")\", team \(teams), expires \(expiry), \(devices) devices")
        if let entitlements = dict["Entitlements"] as? [String: Any] {
            lines.append("  entitlements: " + entitlements.keys.sorted().joined(separator: ", "))
        }
        let certificates = dict["DeveloperCertificates"] as? [Data] ?? []
        var found = false
        for (i, c) in certificates.enumerated() {
            let same = leaf.map { $0 == c } ?? false
            found = found || same
            lines.append("  certificate \(i): \(certificateLine(c))\(same ? "  <- the imported one" : "")")
        }
        if let leaf {
            lines.append(found ? "  the imported certificate IS in the profile"
                               : "  the imported certificate (\(certificateLine(leaf))) is NOT in the profile: the system will refuse what it signs")
        }
        return lines.joined(separator: "\n")
    }

    static func report(pack: URL, leaf: Data) -> String {
        var parts = ["signature diagnosis (the pack did not load):"]
        if let executable = Bundle.main.executablePath {
            parts.append("the app's own signature (accepted by the system):\n" + describe(executable))
        }
        parts.append("the pack's signature:\n" + describe(pack.path))
        parts.append("imported certificate: " + certificateLine(leaf))
        parts.append(profile(leaf: leaf))
        return parts.joined(separator: "\n")
    }
}

