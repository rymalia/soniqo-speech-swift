import CoreML
import Foundation

final class CTCWordBoostingValidator {
    private struct Frame {
        let index: Int
        let logProbs: [Float]
        let bestLogProb: Float
    }

    private let classCount: Int
    private let maxFrames: Int
    private let lookaheadFrames: Int
    private let maxTokenMargin: Float
    private let minTokenLogProb: Float
    private let ignoredTokenIds: Set<Int>
    private var frames: [Frame] = []

    init(
        classCount: Int,
        maxFrames: Int = 32,
        lookaheadFrames: Int = 2,
        maxTokenMargin: Float = 5.0,
        minTokenLogProb: Float = -8.5,
        ignoredTokenIds: Set<Int> = []
    ) {
        self.classCount = classCount
        self.maxFrames = maxFrames
        self.lookaheadFrames = lookaheadFrames
        self.maxTokenMargin = maxTokenMargin
        self.minTokenLogProb = minTokenLogProb
        self.ignoredTokenIds = ignoredTokenIds
    }

    func append(logProbs: MLMultiArray, validFrames: Int, startingAt startFrame: Int) {
        guard validFrames > 0 else { return }
        let shape = logProbs.shape.map(\.intValue)
        guard shape.count >= 3 else { return }

        let time = shape[1]
        let classes = shape[2]
        guard classes >= classCount else { return }

        for t in 0..<min(validFrames, time) {
            var values = [Float](repeating: -.infinity, count: classCount)
            var best = -Float.infinity
            for c in 0..<classCount {
                let value = valueAt(logProbs, timeIndex: t, classIndex: c, classCount: classes)
                values[c] = value
                if value > best { best = value }
            }
            frames.append(Frame(index: startFrame + t, logProbs: values, bestLogProb: best))
        }

        if frames.count > maxFrames {
            frames.removeFirst(frames.count - maxFrames)
        }
    }

    func accepts(matchedTokens tokens: [Int], endingAt frameIndex: Int) -> Bool {
        let tokens = tokens.filter { $0 >= 0 && $0 < classCount && !ignoredTokenIds.contains($0) }
        guard !tokens.isEmpty else { return true }
        guard !frames.isEmpty else { return false }

        let maxEnd = frameIndex + lookaheadFrames
        guard let endOffset = frames.lastIndex(where: { $0.index <= maxEnd }) else {
            return false
        }

        let minStart = frames[endOffset].index - max(maxFrames - 1, tokens.count * 3)
        let startOffset = frames.firstIndex(where: { $0.index >= minStart }) ?? 0
        guard startOffset <= endOffset else { return false }

        var searchOffset = startOffset
        for token in tokens {
            guard let match = bestFrame(for: token, from: searchOffset, through: endOffset) else {
                return false
            }
            let margin = match.frame.bestLogProb - match.tokenLogProb
            guard margin <= maxTokenMargin, match.tokenLogProb >= minTokenLogProb else {
                return false
            }
            searchOffset = match.offset + 1
            if searchOffset > endOffset + 1 { return false }
        }

        return true
    }

    private func bestFrame(
        for token: Int,
        from startOffset: Int,
        through endOffset: Int
    ) -> (offset: Int, frame: Frame, tokenLogProb: Float)? {
        var best: (offset: Int, frame: Frame, tokenLogProb: Float)?

        for offset in startOffset...endOffset {
            let frame = frames[offset]
            let logProb = frame.logProbs[token]
            if let current = best {
                let currentMargin = current.frame.bestLogProb - current.tokenLogProb
                let margin = frame.bestLogProb - logProb
                if margin < currentMargin || (margin == currentMargin && logProb > current.tokenLogProb) {
                    best = (offset, frame, logProb)
                }
            } else {
                best = (offset, frame, logProb)
            }
        }

        return best
    }

    private func valueAt(
        _ array: MLMultiArray,
        timeIndex: Int,
        classIndex: Int,
        classCount: Int
    ) -> Float {
        let strides = array.strides.map(\.intValue)
        let offset: Int
        if strides.count >= 3 {
            offset = timeIndex * strides[1] + classIndex * strides[2]
        } else {
            offset = timeIndex * classCount + classIndex
        }
        switch array.dataType {
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return Float(ptr[offset])
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            return ptr[offset]
        case .double:
            let ptr = array.dataPointer.assumingMemoryBound(to: Double.self)
            return Float(ptr[offset])
        default:
            return Float(truncating: array[[0 as NSNumber, timeIndex as NSNumber, classIndex as NSNumber]])
        }
    }
}
