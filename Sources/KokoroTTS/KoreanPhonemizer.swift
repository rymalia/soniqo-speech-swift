import Foundation

/// Korean text-to-phoneme conversion for Kokoro TTS.
///
/// Pipeline: Korean text → CFStringTransform (Hangul → romanization) → IPA
/// Uses Apple's built-in Korean transliteration — no external dependencies.
final class KoreanPhonemizer {

    // MARK: - Korean Romanization → IPA

    /// Map romanized Korean (from CFStringTransform) to IPA-like phonemes.
    /// Korean romanization from Apple uses Revised Romanization conventions.
    private static let consonantMap: [String: String] = [
        "kk": "k͈", "tt": "t͈", "pp": "p͈", "ss": "s͈", "jj": "tɕ͈",
        "ch": "tɕʰ", "ng": "ŋ",
        "g": "k", "k": "kʰ", "n": "n", "d": "t", "t": "tʰ",
        "r": "ɾ", "l": "l", "m": "m", "b": "p", "p": "pʰ",
        "s": "s", "j": "tɕ", "h": "h",
    ]

    private static let vowelMap: [String: String] = [
        "eo": "ʌ", "eu": "ɯ", "ae": "ɛ", "oe": "we", "wi": "wi",
        "wa": "wa", "wo": "wʌ", "we": "we", "wae": "wɛ",
        "ya": "ja", "yeo": "jʌ", "yo": "jo", "yu": "ju",
        "yae": "jɛ", "ye": "je",
        "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    ]

    // MARK: - Korean Punctuation

    private static let punctuationMap: [Character: String] = [
        "，": ",", "。": ".", "！": "!", "？": "?",
        "、": ",", "；": ";", "：": ":",
    ]

    // MARK: - Public API

    func phonemize(_ text: String) -> String {
        var result = ""
        var lastWasWord = false

        let locale = Locale(identifier: "ko") as CFLocale
        let cfText = text as CFString
        let length = CFStringGetLength(cfText)
        guard length > 0 else { return "" }

        let tokenizer = CFStringTokenizerCreate(nil, cfText, CFRangeMake(0, length),
                                                 kCFStringTokenizerUnitWord, locale)

        var tokens: [(range: NSRange, word: String, reading: String?)] = []
        var tokenResult = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while tokenResult != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let latin = CFStringTokenizerCopyCurrentTokenAttribute(
                tokenizer, kCFStringTokenizerAttributeLatinTranscription) as? String
            let nsRange = NSRange(location: range.location, length: range.length)
            let word = (text as NSString).substring(with: nsRange)
            tokens.append((range: nsRange, word: word, reading: latin))
            tokenResult = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }

        var cursor = 0
        for token in tokens {
            // Handle gaps
            if token.range.location > cursor {
                let gapStart = text.index(text.startIndex, offsetBy: cursor)
                let gapEnd = text.index(text.startIndex, offsetBy: token.range.location)
                for ch in text[gapStart..<gapEnd] {
                    if let punct = Self.punctuationMap[ch] {
                        result += punct
                        lastWasWord = false
                    } else if ch.isPunctuation {
                        result += String(ch)
                        lastWasWord = false
                    } else if ch.isWhitespace {
                        lastWasWord = false
                    }
                }
            }

            if let reading = token.reading {
                if lastWasWord { result += " " }
                result += Self.romanToIPA(reading.lowercased())
                lastWasWord = true
            }

            cursor = token.range.location + token.range.length
        }

        // Trailing punctuation
        if cursor < (text as NSString).length {
            let remaining = (text as NSString).substring(from: cursor)
            for ch in remaining {
                if ch.isPunctuation { result += String(ch) }
            }
        }

        return result
    }

    // MARK: - Romanization → IPA

    static func romanToIPA(_ roman: String) -> String {
        var result = ""
        let chars = Array(roman)
        var i = 0

        while i < chars.count {
            // Skip hyphens (from compound romanization like "igeos-eun")
            if chars[i] == "-" || chars[i] == "'" {
                i += 1
                continue
            }

            // Try 3-char vowel sequences first
            if i + 2 < chars.count {
                let tri = String(chars[i...i+2])
                if let ipa = vowelMap[tri] {
                    result += ipa
                    i += 3
                    continue
                }
            }

            // Try 2-char sequences (consonant clusters or diphthongs)
            if i + 1 < chars.count {
                let pair = String(chars[i...i+1])
                if let ipa = consonantMap[pair] {
                    result += ipa
                    i += 2
                    continue
                }
                if let ipa = vowelMap[pair] {
                    result += ipa
                    i += 2
                    continue
                }
            }

            // Single character
            let single = String(chars[i])
            if let ipa = consonantMap[single] {
                result += ipa
            } else if let ipa = vowelMap[single] {
                result += ipa
            } else {
                result += single
            }
            i += 1
        }

        return result
    }
}
