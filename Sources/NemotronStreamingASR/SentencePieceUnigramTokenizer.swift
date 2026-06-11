import Foundation
import AudioCommon

struct NemotronSentencePieceUnigramTokenizer: Sendable {
    private struct Piece: Sendable {
        let token: String
        let score: Float
        let type: Int
    }

    private let pieces: [Piece]
    private let tokenToId: [String: Int]

    init(modelURL: URL) throws {
        let data = try Data(contentsOf: modelURL)
        let pieces = try Self.parsePieces(from: data)
        guard !pieces.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelURL.lastPathComponent,
                reason: "SentencePiece model did not contain vocabulary pieces"
            )
        }
        self.pieces = pieces
        self.tokenToId = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($0.element.token, $0.offset) })
    }

    init(pieces: [(token: String, score: Float, type: Int)]) {
        self.pieces = pieces.map { Piece(token: $0.token, score: $0.score, type: $0.type) }
        self.tokenToId = Dictionary(uniqueKeysWithValues: self.pieces.enumerated().map { ($0.element.token, $0.offset) })
    }

    func encodeForWordBoosting(_ phrase: String) -> [Int]? {
        let normalized = pieceText(for: phrase)
        guard !normalized.isEmpty else { return nil }

        let scalars = Array(normalized.unicodeScalars)
        var bestScores = [Float](repeating: -.infinity, count: scalars.count + 1)
        var bestPaths = Array(repeating: [Int](), count: scalars.count + 1)
        bestScores[0] = 0

        for start in 0..<scalars.count where bestScores[start].isFinite {
            for end in (start + 1)...scalars.count {
                let token = String(String.UnicodeScalarView(scalars[start..<end]))
                guard let id = tokenToId[token], id != 0 else { continue }

                let candidateScore = bestScores[start] + pieces[id].score
                let candidatePath = bestPaths[start] + [id]
                if candidateScore > bestScores[end]
                    || (candidateScore == bestScores[end] && candidatePath.lexicographicallyPrecedes(bestPaths[end]))
                {
                    bestScores[end] = candidateScore
                    bestPaths[end] = candidatePath
                }
            }
        }

        let path = bestPaths[scalars.count]
        return path.isEmpty ? nil : path
    }

    private func pieceText(for phrase: String) -> String {
        let normalized = (phrase as NSString)
            .precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        guard !normalized.isEmpty else { return "" }
        return "▁" + normalized.replacingOccurrences(of: " ", with: "▁")
    }

    private static func parsePieces(from data: Data) throws -> [Piece] {
        var reader = ProtobufReader(data)
        var pieces: [Piece] = []

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let field = Int(key >> 3)
            let wireType = Int(key & 0x7)

            if field == 1, wireType == 2 {
                let message = try reader.readLengthDelimited()
                pieces.append(try parsePiece(from: message))
            } else {
                try reader.skip(wireType: wireType)
            }
        }

        return pieces
    }

    private static func parsePiece(from data: Data) throws -> Piece {
        var reader = ProtobufReader(data)
        var token = ""
        var score: Float = 0
        var type = 1

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let field = Int(key >> 3)
            let wireType = Int(key & 0x7)

            switch (field, wireType) {
            case (1, 2):
                token = String(data: try reader.readLengthDelimited(), encoding: .utf8) ?? ""
            case (2, 5):
                score = try reader.readFloat32()
            case (3, 0):
                type = Int(try reader.readVarint())
            default:
                try reader.skip(wireType: wireType)
            }
        }

        return Piece(token: token, score: score, type: type)
    }
}

private struct ProtobufReader {
    private let bytes: [UInt8]
    private var offset = 0

    var isAtEnd: Bool { offset >= bytes.count }

    init(_ data: Data) {
        self.bytes = Array(data)
    }

    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0

        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
            if shift >= 64 { break }
        }

        throw AudioModelError.modelLoadFailed(modelId: "tokenizer.model", reason: "Malformed protobuf varint")
    }

    mutating func readLengthDelimited() throws -> Data {
        let length = Int(try readVarint())
        guard offset + length <= bytes.count else {
            throw AudioModelError.modelLoadFailed(modelId: "tokenizer.model", reason: "Malformed protobuf length")
        }
        defer { offset += length }
        return Data(bytes[offset..<(offset + length)])
    }

    mutating func readFloat32() throws -> Float {
        guard offset + 4 <= bytes.count else {
            throw AudioModelError.modelLoadFailed(modelId: "tokenizer.model", reason: "Malformed protobuf float")
        }
        let bits = UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
        offset += 4
        return Float(bitPattern: bits)
    }

    mutating func skip(wireType: Int) throws {
        switch wireType {
        case 0:
            _ = try readVarint()
        case 1:
            offset += 8
        case 2:
            _ = try readLengthDelimited()
        case 5:
            offset += 4
        default:
            throw AudioModelError.modelLoadFailed(
                modelId: "tokenizer.model",
                reason: "Unsupported protobuf wire type \(wireType)"
            )
        }

        guard offset <= bytes.count else {
            throw AudioModelError.modelLoadFailed(modelId: "tokenizer.model", reason: "Malformed protobuf skip")
        }
    }
}
