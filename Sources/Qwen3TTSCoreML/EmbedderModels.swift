#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML text token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side text_embedding + FC1→SiLU→FC2 projection.
final class TextProjectorModel {
    private let model: MLModel
    init(model: MLModel) { self.model = model }

    func embed(_ tokenId: Int) throws -> MLMultiArray {
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(tokenId)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

/// CoreML codec token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side codec_embedding table lookup.
final class CodeEmbedderModel {
    private let model: MLModel
    init(model: MLModel) { self.model = model }

    func embed(_ tokenId: Int) throws -> MLMultiArray {
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(tokenId)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

/// CoreML linearized CB1-15 token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side cpCodecEmbeddings table lookup + sum.
final class MultiCodeEmbedderModel {
    private let model: MLModel
    private let vocabSize = 2048
    init(model: MLModel) { self.model = model }

    /// Embed a single codebook token using linearized index: codebookIdx * 2048 + tokenId.
    func embed(codebookIdx: Int, tokenId: Int) throws -> MLMultiArray {
        let linearIdx = codebookIdx * vocabSize + tokenId
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(linearIdx)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

// MARK: - MLMultiArray helpers

/// Add two MLMultiArrays element-wise. Both must be the same shape and FP16.
func addMLMultiArrays(_ a: MLMultiArray, _ b: MLMultiArray) -> MLMultiArray {
    let count = a.shape.map { $0.intValue }.reduce(1, *)
    let result = try! MLMultiArray(shape: a.shape, dataType: .float16)
    let ap = a.dataPointer.assumingMemoryBound(to: Float16.self)
    let bp = b.dataPointer.assumingMemoryBound(to: Float16.self)
    let rp = result.dataPointer.assumingMemoryBound(to: Float16.self)
    for i in 0..<count {
        rp[i] = Float16(Float(ap[i]) + Float(bp[i]))
    }
    return result
}

/// Ensure MLMultiArray has rank 4 [1, C, 1, 1] (CoreML sometimes drops batch dim).
func ensureNCHW(_ array: MLMultiArray, channels: Int) -> MLMultiArray {
    if array.shape.count == 4 { return array }
    let result = try! MLMultiArray(shape: [1, NSNumber(value: channels), 1, 1], dataType: array.dataType)
    let bytes = channels * (array.dataType == .float16 ? 2 : 4)
    memcpy(result.dataPointer, array.dataPointer, bytes)
    return result
}
#endif
