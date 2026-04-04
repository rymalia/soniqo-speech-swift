import Foundation

/// Grapheme-to-phoneme conversion for Latin-script languages (French, Spanish, Portuguese).
///
/// Rule-based orthography→IPA conversion. Each language has specific rules for
/// digraphs, accent handling, and context-dependent pronunciation.
/// No external dependencies — pure Swift string processing.
final class LatinPhonemizer {

    enum Language {
        case french, spanish, portuguese
    }

    private let language: Language

    init(language: Language) {
        self.language = language
    }

    // MARK: - Public API

    func phonemize(_ text: String) -> String {
        let words = tokenize(text)
        var result = ""
        var lastWasWord = false

        for token in words {
            switch token {
            case .word(let w):
                if lastWasWord { result += " " }
                result += convertWord(w.lowercased())
                lastWasWord = true
            case .punctuation(let p):
                result += p
                lastWasWord = false
            case .space:
                lastWasWord = false
            }
        }

        return result
    }

    // MARK: - Tokenization

    private enum Token {
        case word(String)
        case punctuation(String)
        case space
    }

    private func tokenize(_ text: String) -> [Token] {
        var tokens: [Token] = []
        var current = ""

        for ch in text {
            if ch.isWhitespace {
                if !current.isEmpty { tokens.append(.word(current)); current = "" }
                tokens.append(.space)
            } else if ch.isLetter || ch == "'" || ch == "'" || ch == "-" {
                current.append(ch)
            } else if ch.isPunctuation || ch.isSymbol {
                if !current.isEmpty { tokens.append(.word(current)); current = "" }
                tokens.append(.punctuation(String(ch)))
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { tokens.append(.word(current)) }

        return tokens
    }

    // MARK: - Word Conversion

    private func convertWord(_ word: String) -> String {
        switch language {
        case .french: return frenchToIPA(word)
        case .spanish: return spanishToIPA(word)
        case .portuguese: return portugueseToIPA(word)
        }
    }

    // MARK: - French G2P

    /// French grapheme-to-phoneme rules.
    private static let frenchRules: [(pattern: String, ipa: String)] = [
        // Trigraphs / special combos
        ("eau", "o"), ("aux", "o"), ("eux", "ø"), ("oeu", "œ"),
        ("ain", "ɛ̃"), ("ein", "ɛ̃"), ("oin", "wɛ̃"),
        ("ien", "jɛ̃"), ("ion", "jɔ̃"),
        // Nasal vowels
        ("an", "ɑ̃"), ("am", "ɑ̃"), ("en", "ɑ̃"), ("em", "ɑ̃"),
        ("on", "ɔ̃"), ("om", "ɔ̃"), ("un", "œ̃"), ("um", "œ̃"),
        ("in", "ɛ̃"), ("im", "ɛ̃"),
        // Digraphs
        ("ou", "u"), ("oi", "wa"), ("ai", "ɛ"), ("ei", "ɛ"),
        ("au", "o"), ("eu", "ø"), ("ch", "ʃ"), ("ph", "f"),
        ("th", "t"), ("gn", "ɲ"), ("qu", "k"), ("gu", "ɡ"),
        ("ll", "l"), ("ss", "s"), ("tt", "t"), ("nn", "n"),
        ("mm", "m"), ("pp", "p"), ("rr", "ʁ"), ("ff", "f"),
        // Accented vowels
        ("é", "e"), ("è", "ɛ"), ("ê", "ɛ"), ("ë", "ɛ"),
        ("à", "a"), ("â", "ɑ"), ("ù", "y"), ("û", "y"),
        ("î", "i"), ("ï", "i"), ("ô", "o"), ("ü", "y"),
        ("ç", "s"), ("œ", "œ"),
        // Basic
        ("a", "a"), ("b", "b"), ("c", "k"), ("d", "d"), ("e", "ə"),
        ("f", "f"), ("g", "ɡ"), ("h", ""), ("i", "i"), ("j", "ʒ"),
        ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("o", "o"),
        ("p", "p"), ("r", "ʁ"), ("s", "s"), ("t", "t"), ("u", "y"),
        ("v", "v"), ("w", "w"), ("x", "ks"), ("y", "i"), ("z", "z"),
    ]

    private func frenchToIPA(_ word: String) -> String {
        var result = ""
        let chars = Array(word)
        var i = 0

        while i < chars.count {
            var matched = false
            // Try longest match first (3, 2, 1 chars)
            for len in stride(from: min(3, chars.count - i), through: 1, by: -1) {
                let substr = String(chars[i..<i+len])
                if let rule = Self.frenchRules.first(where: { $0.pattern == substr }) {
                    result += rule.ipa
                    i += len
                    matched = true
                    break
                }
            }
            if !matched {
                result += String(chars[i])
                i += 1
            }
        }

        // Drop silent final consonants (simplified French rule)
        if result.count > 1 {
            let last = result.last!
            if "dtsxzp".contains(last) && word.last != "c" {
                result = String(result.dropLast())
            }
        }

        return result
    }

    // MARK: - Spanish G2P

    /// Spanish is very regular — nearly 1:1 grapheme-to-phoneme.
    private static let spanishRules: [(pattern: String, ipa: String)] = [
        // Digraphs
        ("ch", "tʃ"), ("ll", "ʝ"), ("rr", "r"), ("qu", "k"),
        ("gu", "ɡ"), ("gü", "ɡw"),
        ("ñ", "ɲ"),
        // Accented vowels (same sound, just stress)
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ü", "w"),
        // Basic
        ("a", "a"), ("b", "b"), ("c", "k"), ("d", "d"), ("e", "e"),
        ("f", "f"), ("g", "ɡ"), ("h", ""), ("i", "i"), ("j", "x"),
        ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("o", "o"),
        ("p", "p"), ("r", "ɾ"), ("s", "s"), ("t", "t"), ("u", "u"),
        ("v", "b"), ("w", "w"), ("x", "ks"), ("y", "ʝ"), ("z", "θ"),
    ]

    private func spanishToIPA(_ word: String) -> String {
        var result = ""
        let chars = Array(word)
        var i = 0

        while i < chars.count {
            var matched = false
            for len in stride(from: min(2, chars.count - i), through: 1, by: -1) {
                let substr = String(chars[i..<i+len])
                if let rule = Self.spanishRules.first(where: { $0.pattern == substr }) {
                    // Context: c before e/i = θ, g before e/i = x
                    if substr == "c" && i + 1 < chars.count && "eiéí".contains(chars[i+1]) {
                        result += "θ"
                    } else if substr == "g" && i + 1 < chars.count && "eiéí".contains(chars[i+1]) {
                        result += "x"
                    } else {
                        result += rule.ipa
                    }
                    i += len
                    matched = true
                    break
                }
            }
            if !matched {
                result += String(chars[i])
                i += 1
            }
        }

        return result
    }

    // MARK: - Portuguese G2P

    private static let portugueseRules: [(pattern: String, ipa: String)] = [
        // Digraphs / trigraphs
        ("ção", "sɐ̃w̃"), ("ções", "sɔ̃j̃s"), ("nh", "ɲ"), ("lh", "ʎ"),
        ("ch", "ʃ"), ("qu", "k"), ("gu", "ɡ"), ("rr", "ʁ"),
        ("ss", "s"), ("sc", "s"),
        // Nasal
        ("ão", "ɐ̃w̃"), ("ãe", "ɐ̃j̃"), ("õe", "õj̃"),
        ("an", "ɐ̃"), ("am", "ɐ̃"), ("en", "ẽ"), ("em", "ẽ"),
        ("in", "ĩ"), ("im", "ĩ"), ("on", "õ"), ("om", "õ"),
        ("un", "ũ"), ("um", "ũ"),
        // Accented
        ("á", "a"), ("â", "ɐ"), ("ã", "ɐ̃"), ("é", "ɛ"), ("ê", "e"),
        ("í", "i"), ("ó", "ɔ"), ("ô", "o"), ("õ", "õ"), ("ú", "u"),
        ("ç", "s"),
        // Diphthongs
        ("ou", "o"), ("ei", "ej"), ("ai", "aj"), ("oi", "oj"),
        // Basic
        ("a", "a"), ("b", "b"), ("c", "k"), ("d", "d"), ("e", "e"),
        ("f", "f"), ("g", "ɡ"), ("h", ""), ("i", "i"), ("j", "ʒ"),
        ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("o", "o"),
        ("p", "p"), ("r", "ɾ"), ("s", "s"), ("t", "t"), ("u", "u"),
        ("v", "v"), ("w", "w"), ("x", "ʃ"), ("y", "i"), ("z", "z"),
    ]

    private func portugueseToIPA(_ word: String) -> String {
        var result = ""
        let chars = Array(word)
        var i = 0

        while i < chars.count {
            var matched = false
            for len in stride(from: min(4, chars.count - i), through: 1, by: -1) {
                let substr = String(chars[i..<i+len])
                if let rule = Self.portugueseRules.first(where: { $0.pattern == substr }) {
                    result += rule.ipa
                    i += len
                    matched = true
                    break
                }
            }
            if !matched {
                result += String(chars[i])
                i += 1
            }
        }

        return result
    }
}
