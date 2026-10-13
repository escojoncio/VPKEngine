// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// VPKEngine demo: runs an x86-64 program that vpaot translated to C (compiled into this app as
// arm64 code) and checks its result against the same program compiled natively.

import SwiftUI

@main
struct DemoApp: App {
    var body: some Scene {
        WindowGroup {
            DemoView()
        }
        .defaultSize(width: 720, height: 480)
    }
}

struct DemoView: View {
    @State private var message = "Pulsa Ejecutar para correr el programa x86-64 traducido."
    @State private var detail = ""
    @State private var ok: Bool? = nil
    @State private var running = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("VPKEngine").font(.largeTitle).bold()
            Text("Traducción anticipada x86-64 → C → arm64, sin JIT.").foregroundStyle(.secondary)
            Divider()
            Text(message).font(.body)
            if !detail.isEmpty { Text(detail).font(.callout.monospaced()).foregroundStyle(.secondary) }
            Spacer()
            HStack {
                Button(running ? "Ejecutando…" : "Ejecutar") { run() }
                    .disabled(running)
                    .buttonStyle(.borderedProminent)
                if let ok {
                    Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(ok ? .green : .red)
                        .font(.title)
                }
            }
        }
        .padding(32)
    }

    private func run() {
        running = true
        ok = nil
        DispatchQueue.global().async {
            let path = Bundle.main.path(forResource: "prog", ofType: "elf") ?? ""
            var result = vp_demo_run(path)
            let text = withUnsafePointer(to: &result.message) { p in
                p.withMemoryRebound(to: CChar.self, capacity: 512) { String(cString: $0) }
            }
            let timing = String(format: "traducido %.3f ms · nativo %.3f ms", result.translated_ms, result.native_ms)
            DispatchQueue.main.async {
                message = text
                detail = result.ok != 0 ? timing : ""
                ok = result.ok != 0
                running = false
            }
        }
    }
}
