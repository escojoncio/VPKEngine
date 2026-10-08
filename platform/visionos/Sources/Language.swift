// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// The app's language, chosen in Ajustes / Settings: Spanish, English, or the system's (Spanish
// when the system's first language is Spanish, English otherwise). Every text the app shows is
// written in both, side by side, as L("español", "English"); views redraw when the choice
// changes. (The texts visionOS shows for the app's permission requests follow the system's
// language: es.lproj and en.lproj InfoPlist.strings.)

import Foundation
import Observation

enum AppLanguage: String, CaseIterable, Identifiable {
    case system
    case spanish = "es"
    case english = "en"

    var id: String { rawValue }
}

@Observable
final class Language: @unchecked Sendable {
    static let shared = Language()

    private static let key = "pt.language"

    var choice: AppLanguage {
        didSet {
            UserDefaults.standard.set(choice.rawValue, forKey: Self.key)
        }
    }

    private init() {
        choice = AppLanguage(rawValue: UserDefaults.standard.string(forKey: Self.key) ?? "") ?? .system
    }

    var isSpanish: Bool {
        switch choice {
        case .spanish:
            return true
        case .english:
            return false
        case .system:
            return (Locale.preferredLanguages.first ?? "en").lowercased().hasPrefix("es")
        }
    }

    /// "es" or "en", whatever the choice: what the game is told.
    var code: String {
        isSpanish ? "es" : "en"
    }

    var locale: Locale {
        Locale(identifier: isSpanish ? "es_ES" : "en_US")
    }
}

/// The text in the app's language.
func L(_ spanish: String, _ english: String) -> String {
    Language.shared.isSpanish ? spanish : english
}
