import AppKit
import Combine
import Foundation
import ParakeetStreamingASR
import SpeechVAD

private let logPath = "/tmp/dictate.log"
private let logLock = NSLock()
private func dlog(_ msg: String) {
    logLock.lock()
    defer { logLock.unlock() }
    if let data = "\(msg)\n".data(using: .utf8) {
        if let fh = FileHandle(forWritingAtPath: logPath) {
            fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
        } else {
            FileManager.default.createFile(atPath: logPath, contents: data)
        }
    }
}

/// Off-main-thread audio processing.
final class ASRProcessor: Sendable {
    let session: StreamingSession
    private let vad: SileroVADModel
    private let lock = NSLock()
    private let _buffer = UnsafeMutablePointer<[Float]>.allocate(capacity: 1)
    nonisolated(unsafe) var speechActive = false

    init(session: StreamingSession, vad: SileroVADModel) {
        self.session = session
        self.vad = vad
        _buffer.initialize(to: [])
    }
    deinit { _buffer.deinitialize(count: 1); _buffer.deallocate() }

    func appendAudio(_ samples: [Float]) {
        lock.lock()
        _buffer.pointee.append(contentsOf: samples)
        lock.unlock()
    }

    var bufferedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return _buffer.pointee.count
    }

    func processBuffered() -> (partials: [ParakeetStreamingASRModel.PartialTranscript], speaking: Bool) {
        lock.lock()
        let chunk = _buffer.pointee
        _buffer.pointee.removeAll(keepingCapacity: true)
        lock.unlock()
        guard !chunk.isEmpty else { return ([], speechActive) }

        // VAD
        var offset = 0
        var silenceCount = 0
        while offset + 512 <= chunk.count {
            let prob = vad.processChunk(Array(chunk[offset..<offset+512]))
            if prob >= 0.5 { speechActive = true; silenceCount = 0 }
            else { silenceCount += 1; if silenceCount >= 15 { speechActive = false } }
            offset += 512
        }

        // ASR
        do {
            let partials = try session.pushAudio(chunk)
            return (partials, speechActive)
        } catch {
            dlog("ASR error: \(error)")
            return ([], speechActive)
        }
    }

    func finalize() -> [ParakeetStreamingASRModel.PartialTranscript] {
        let (r, _) = processBuffered()
        do { return r + (try session.finalize()) }
        catch { return r }
    }
}

@MainActor
final class DictateViewModel: ObservableObject {
    @Published var sentences: [String] = []
    @Published var partialText = ""
    @Published var isRecording = false
    @Published var isLoading = false
    @Published var loadingStatus = ""
    @Published var errorMessage: String?
    @Published var isSpeechActive = false
    @Published var shouldShowHUD = false

    private var model: ParakeetStreamingASRModel?
    private var vad: SileroVADModel?
    private var processor: ASRProcessor?
    private let recorder = StreamingRecorder()
    private let processQueue = DispatchQueue(label: "dictate.asr", qos: .userInteractive)
    private var processTimer: DispatchSourceTimer?

    var modelLoaded: Bool { model != nil && vad != nil }
    var audioLevel: Float { recorder.audioLevel }

    var wordCount: Int {
        let all = sentences.joined(separator: " ") + (partialText.isEmpty ? "" : " " + partialText)
        return all.split(separator: " ").count
    }

    var fullText: String {
        let committed = sentences.joined(separator: "\n")
        if committed.isEmpty { return partialText }
        if partialText.isEmpty { return committed }
        return committed + "\n" + partialText
    }

    init() {
        Task { await loadModels() }
    }

    func loadModels() async {
        guard model == nil else { return }
        isLoading = true
        loadingStatus = "Downloading ASR model..."

        do {
            let loaded = try await Task.detached {
                try await ParakeetStreamingASRModel.fromPretrained { [weak self] p, s in
                    DispatchQueue.main.async {
                        self?.loadingStatus = s.isEmpty ? "Downloading... \(Int(p * 100))%" : "\(s) (\(Int(p * 100))%)"
                    }
                }
            }.value
            loadingStatus = "Warming up..."
            try loaded.warmUp()
            model = loaded

            loadingStatus = "Loading VAD..."
            vad = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml)
            }.value
            loadingStatus = ""
            dlog("Models loaded (ASR + VAD)")
        } catch {
            errorMessage = "Failed: \(error.localizedDescription)"
            loadingStatus = ""
        }
        isLoading = false
    }

    func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    func startRecording() {
        guard let model, let vad else { return }
        errorMessage = nil; partialText = ""; sentences.removeAll()

        do {
            let session = try model.createSession()
            let proc = ASRProcessor(session: session, vad: vad)
            processor = proc

            recorder.start { [proc] chunk in proc.appendAudio(chunk) }

            let timer = DispatchSource.makeTimerSource(queue: processQueue)
            timer.schedule(deadline: .now(), repeating: .milliseconds(300))
            timer.setEventHandler { [weak self, proc] in
                let (partials, speaking) = proc.processBuffered()
                DispatchQueue.main.async {
                    self?.isSpeechActive = speaking
                    for partial in partials {
                        if partial.isFinal && !partial.text.isEmpty {
                            dlog("FINAL: '\(partial.text)'")
                            self?.sentences.append(partial.text)
                            self?.partialText = ""
                        } else if !partial.text.isEmpty {
                            self?.partialText = partial.text
                        }
                    }
                }
            }
            timer.resume()
            processTimer = timer
            isRecording = true
            shouldShowHUD = true
            dlog("Recording started")
        } catch {
            errorMessage = "Failed: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        processTimer?.cancel(); processTimer = nil
        recorder.stop(); isRecording = false; isSpeechActive = false
        if let processor {
            for p in processor.finalize() where !p.text.isEmpty { sentences.append(p.text) }
        }
        processor = nil; partialText = ""
    }

    func pasteToFrontApp() {
        guard !fullText.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let src = CGEventSource(stateID: .hidSystemState)
            let kd = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: true); kd?.flags = .maskCommand
            let ku = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: false); ku?.flags = .maskCommand
            kd?.post(tap: .cghidEventTap); ku?.post(tap: .cghidEventTap)
        }
    }

    func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
    }

    func clearText() { sentences.removeAll(); partialText = "" }
}
