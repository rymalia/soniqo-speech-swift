---
date: 2026-03-30
time: "10:48 PM – 12:20 AM PDT"
project: soniqo-speech-swift
branch: main
---

# Session Summary: Codebase Deep Dive & Memory Bootstrap

## Objective

First session on this project. Full codebase exploration to build understanding of the architecture, module graph, model inventory, test patterns, CI/CD pipeline, and developer conventions. Bootstrapped persistent memory and local agent config.

## Changes Made

| Change | Detail |
|--------|--------|
| **Created CLAUDE.local.md** | Quick-reference local agent config: module tiers, test cheat sheet, build notes, format conventions, doc update checklist |
| **Bootstrapped memory system** | 6 memory files + MEMORY.md index covering user profile, architecture, model inventory, git conventions, README translations, documentation references |

## Exploration Scope

4 parallel explorer agents covered the full codebase:

- **151 Swift source files** (~34K LOC) across 17 SPM targets
- **44 test files** (~13K LOC) across 13 test targets
- **12+ Python conversion scripts**, 5 benchmark scripts, build/CI scripts
- **2 demo apps** (SpeechDemo, PersonaPlexDemo), WebSocket server, Homebrew formula

---

## Architecture Deep Dive

### Module Dependency Graph

```
Tier 0 (Foundation)
  AudioCommon ──→ MLXCommon
      │                │
Tier 1 (Models)        │
  ┌────────────────────┘
  │  MLX-based:                     CoreML-based:
  │  ├─ Qwen3ASR (0.6B/1.7B)       ├─ ParakeetASR (INT4/INT8)
  │  ├─ Qwen3TTS (0.6B/1.7B)       ├─ KokoroTTS (82M)
  │  ├─ CosyVoiceTTS (0.5B)        └─ SpeechEnhancement (DeepFilterNet3)
  │  ├─ PersonaPlex (7B)
  │  ├─ SpeechVAD (Pyannote/Silero/FireRedVAD)
  │  └─ Qwen3Chat (0.8B)
  │
Tier 2 (Apps)
  ├─ AudioCLILib (aggregates all models, 11 CLI commands)
  └─ AudioServer (Hummingbird HTTP/WS, OpenAI Realtime API compat)

Tier 3 (Entry Points)
  ├─ AudioCLI (executable)
  └─ AudioServerCLI (executable)
```

### Dual-Backend Strategy

The project uses two inference backends behind shared protocol interfaces:

- **MLX (Metal GPU)**: For autoregressive models needing flexible generation (ASR decoders, TTS, LLM, speech-to-speech). Weights in safetensors with 4-bit/8-bit quantized decoders, FP16 encoders.
- **CoreML (Neural Engine)**: For fixed-shape models targeting iOS power efficiency (Parakeet ASR, Kokoro TTS, DeepFilterNet3, Silero VAD). Weights in mlpackage format.

Modules that only depend on `AudioCommon` (not `MLXCommon`) are pure CoreML — a clean architectural boundary.

---

## Foundation Layer (AudioCommon + MLXCommon)

### AudioCommon (12 files, ~1,800 LOC)

**Protocols** (`Protocols.swift`, 268 lines): Every model capability is a protocol:
- `SpeechRecognitionModel` — ASR (blocking + async)
- `SpeechGenerationModel` — TTS (sync + streaming via `AsyncThrowingStream<AudioChunk>`)
- `SpeechToSpeechModel` — full-duplex speech-to-speech
- `VoiceActivityDetectionModel` / `StreamingVADProvider` — VAD (batch + streaming 32ms chunks)
- `SpeakerEmbeddingModel` / `SpeakerDiarizationModel` — speaker ID and diarization
- `SpeechEnhancementModel` — noise suppression
- `ForcedAlignmentModel` — word-level timestamps
- `ModelMemoryManageable` — explicit GPU memory management (`unload()`, `memoryFootprint`)
- `PipelineLLM` — LLM bridge for voice pipeline (ASR→LLM→TTS flow)

**Streaming Audio** (`StreamingAudioPlayer.swift`, 472 lines): Production-grade TTS playback engine:
- SPSC ring buffer architecture: TTS producer → ring buffer → AVAudioSourceNode render callback
- Pre-buffer threshold (default 1.0s) for latency-jitter tradeoff
- Cross-fade at chunk boundaries (10ms fade, 480 samples at 48kHz)
- Warmup chunk filtering (RMS < 0.02), silence compression
- Resampling support via cached AVAudioConverter

**Audio I/O** (`AudioIO.swift`, 174 lines): Microphone capture + playback with hardware echo cancellation, auto-resampling to 16kHz, RMS level metering.

**Ring Buffers**: Two ring buffer implementations — `AudioRingBuffer` (capture→inference) and `AudioSampleRingBuffer` (TTS→playback). Both use drop-oldest semantics with NSLock for thread safety.

**HuggingFace Downloader** (`HuggingFaceDownloader.swift`, 199 lines): Unified model download/caching with dual-mode cache paths (old flat → new Hub-style with auto-migration), path traversal attack prevention, environment variable overrides.

**Tokenizer** (`Tokenizer.swift`, 329 lines): Qwen3 BPE tokenizer with GPT-2-style byte-to-unicode mapping, handles multi-byte CJK correctly.

**Model Loader** (`ModelLoader.swift`, 176 lines): Coordinated multi-model loading with parallel group 0 (VAD + ASR) and sequential group 1 (TTS) to limit peak GPU memory.

### MLXCommon (4 files, ~350 LOC)

- `CommonWeightLoader` — Generic safetensors loading with prefix-based slicing, handles conv1d PyTorch→MLX transpose
- `PreQuantizedEmbedding` — 4-bit embedding layer with per-group scales/biases
- `QuantizedMLP` — SwiGLU MLP (gate + up + down) shared across ASR/TTS/code predictor
- `ModuleMemory` — Parameter introspection and GPU memory cleanup

---

## Model Modules

### Qwen3ASR (17 files) — MLX Speech-to-Text

- **Audio Encoder**: Whisper-style mel spectrogram (128 bins, vDSP FFT) → transformer encoder (18 or 24 layers)
- **Text Decoder**: Quantized transformer with GQA (16 heads, 8 KV heads), RoPE (theta=1M), tied embedding weights
- **Inference**: Mel features → audio encoder → embed with special tokens → autoregressive text generation with KV cache
- **Forced Alignment**: Word-level timestamps via alignment model + LIS timestamp correction
- **Streaming**: StreamingASR combines ASR + Silero VAD, segments by speech boundaries

### Qwen3TTS (14 files) — MLX Text-to-Speech

- **Three-stage pipeline**: Talker (text→speech tokens, multi-position RoPE) → Code Predictor (16 codebook groups, 5-layer transformer) → Speech Tokenizer Decoder (codec→mel→waveform)
- **Compilation**: Talker step compiled shapeless; Code Predictor compiled per-cache-size
- **Streaming**: Configurable chunking via StreamingConfig
- **Voice cloning**: In-context learning from reference audio samples

### CosyVoiceTTS (13 files) — MLX Flow-Based TTS

- **Fundamentally different from Qwen3TTS**: Uses flow matching (not autoregressive)
- **Pipeline**: Qwen2.5-0.5B LLM → DiT (22-layer, AdaLayerNormZero, 10 ODE steps) → HiFi-GAN vocoder
- **Dialogue support**: Multi-speaker conversation synthesis via DialogueParser/Synthesizer
- **Speaker**: CAM++ speaker encoder for voice conditioning

### KokoroTTS (6 files) — CoreML iOS TTS

- **82M parameters**, non-autoregressive (fastest TTS option)
- **Phonemizer**: Text → phoneme tokens as input
- **Model buckets**: Selects CoreML model by input length (5s, 10s, 15s, 30s)
- **50 voice presets** via 256-dim style embeddings
- **Runs on Neural Engine** — designed for iOS power efficiency

### ParakeetASR (7 files) — CoreML Transducer ASR

- **FastConformer encoder** + TDT (Token-and-Duration Transducer) decoder
- **INT4/INT8 quantized** CoreML model
- **25 European languages** (auto-detected)
- **Mel preprocessing** via Swift/Accelerate (vDSP), padded to avoid BNNS crash

### PersonaPlex (16 files) — MLX Full-Duplex Speech-to-Speech

- **7B temporal transformer** processing 17 streams (1 text + 8 user audio + 8 agent audio codebooks)
- **Mimi audio codec**: 8 codebooks, variable rate (320–960 bps per codebook)
- **Depformer**: Generates agent audio codebooks
- **Ring-buffer KV cache** with per-stream delays for genuine full-duplex
- **SentencePiece tokenizer** (32K vocab)
- **18 voice presets** with cached temporal state

### SpeechVAD (31 files) — VAD, Diarization, Speaker ID

- **Pyannote**: SincNet (3 filter layers) → BiLSTM (4 layers, 128d) → powerset decoder, hysteresis thresholding
- **Silero**: Streaming 32ms chunks, state machine (silence→pendingSpeech→speech→pendingSilence)
- **FireRedVAD**: DFSMN-based, tunable thresholds
- **WeSpeaker**: Speaker embedding extraction for clustering-based diarization
- **Sortformer**: End-to-end transformer diarizer
- **DER scoring**: Full diarization error rate computation with collar forgiveness and optimal speaker mapping

### SpeechEnhancement (7 files) — CoreML + CPU Hybrid Denoiser

- **DeepFilterNet3**: Neural network (CoreML FP16 on Neural Engine) for ERB band gains
- **CPU DSP**: STFT (960-point, 481 bins) → ERB filterbank → gain application → iSTFT
- **48kHz native**, resamples if needed

### Qwen3Chat (8 files) — MLX On-Device LLM

- **Qwen3.5-0.8B** with hybrid attention architecture
- **DeltaNet (linear attention)**: Recurrent state evolution `S_{t+1} = g * S_t + β * (v_t ⊗ k_t)` — O(n) instead of O(n²)
- **Gated full attention** every 4th layer for quality
- **4-bit quantized** (group_size=64), 248K vocab

### SpeechCore (1 file, ~850 LOC) — Pipeline Orchestration

- **VoicePipeline state machine**: idle → listening → transcribing → thinking → speaking
- **Modes**: Full voice pipeline (VAD→STT→LLM→TTS), transcribe-only, echo (STT→TTS for testing)
- **C bindings**: VTable pattern for bridging STT/TTS models

---

## CLI & Server

### AudioCLI (11 subcommands)

| Command | Description |
|---------|-------------|
| `transcribe` | ASR with engine selection (qwen3/parakeet/qwen3-coreml), streaming support |
| `transcribe-batch` | Batch directory transcription with JSONL output |
| `align` | Forced alignment: word-level timestamps |
| `speak` | TTS with engine/voice/language selection, batch mode |
| `respond` | Speech-to-speech via PersonaPlex (18 voices, system prompts) |
| `vad` | Voice activity detection (pyannote/firered), JSON/RTTM output |
| `vad-stream` | Streaming VAD (Silero, 32ms chunks) |
| `diarize` | Speaker diarization (pyannote/sortformer), DER scoring |
| `embed-speaker` | Speaker embeddings (WeSpeaker/CAM++) |
| `denoise` | Speech enhancement (DeepFilterNet3) |
| `kokoro` | Kokoro TTS (CoreML, 50 voices) |

### AudioServer (Hummingbird HTTP/WS)

- REST endpoints: `/health`, `/transcribe`, `/speak`, `/respond`, `/enhance`
- **WebSocket `/v1/realtime`**: OpenAI Realtime API compatible — session management, base64 PCM16 audio, streaming TTS
- Lazy model loading (or `--preload` for eager startup)

### Demo Apps

- **SpeechDemo**: SwiftUI tabs for Dictate (ASR), Speak (TTS), Echo (S2S)
- **PersonaPlexDemo**: Full-duplex voice assistant with voice presets and system prompts
- **websocket-client.html**: Browser client for the WebSocket server

---

## Test Suite

### Statistics

- **44 test files**, ~13,357 LOC across 13 test targets
- **100% XCTest** (no Swift Testing/@Test)
- **~23 E2E test classes** (prefixed with `E2E`, require GPU + model downloads)
- **~18+ unit test classes** run in CI (explicit filter in Makefile)

### CI Test Strategy

The Makefile `test` target uses an explicit `--filter` string listing every CI-safe test class/method. New unit tests must be manually added to this filter. CI (GitHub Actions, macOS 15 M-series) skips all E2E tests.

### Notable Test Patterns

- **Dynamic WAV builder**: `buildWAV(sampleRate:numChannels:bitsPerSample:samples:)` for security hardening tests
- **Mock-based DI**: `MockVAD`, `MockSTT`, `MockTTS` for model loader tests
- **RTTM round-trip**: Serialize → parse → validate for diarization scoring
- **Known I/O pairs**: Reflection padding `[1,2,3,4,5]` → `[3,2,1,2,3,4,5,4,3]`
- **Environment-driven**: Weight loading tests use env vars (`COSYVOICE_WEIGHTS`), skip if unavailable
- **Async patterns**: XCTestExpectation for streaming playback, Task-based server lifecycle

---

## CI/CD Pipeline

- **tests.yml**: macOS 15 M-series, SPM cache, debug build + metallib, unit tests only, demo app compilation check
- **release.yml**: Release build → package binary + metallib → upload to GitHub Releases → auto-update Homebrew formula SHA/URL → commit back to main

---

## Distribution

- **Homebrew**: `brew tap soniqo/speech && brew install speech` — formula auto-updated by release CI
- **GitHub Releases**: `audio-macos-arm64.tar.gz` containing `audio` binary + `mlx.metallib`
- **SPM libraries**: Each model module is a separate library product for framework consumers

---

## Files Created

| Change | Detail |
|--------|--------|
| **CLAUDE.local.md** | Local agent quick-reference: module tiers, test cheat sheet, build notes, format conventions |
| **memory/user_profile.md** | Developer profile, expertise, preferences |
| **memory/project_architecture.md** | Module graph, dual MLX/CoreML backend strategy |
| **memory/project_model_inventory.md** | Complete model table with backends, sizes, architectures |
| **memory/feedback_git_conventions.md** | No AI mentions, no auto-commits, manual workflow |
| **memory/feedback_readme_translations.md** | 9-translation sync requirement |
| **memory/reference_docs_and_web.md** | Local docs, soniqo.audio site, HF models, CI locations |
| **memory/MEMORY.md** | Index of all memory files |

## Key Architectural Observations

1. **Protocol-first design** enables clean backend swapping (MLX ↔ CoreML) behind shared interfaces
2. **Three distinct TTS paradigms** coexist: autoregressive (Qwen3), flow matching (CosyVoice), non-autoregressive (Kokoro)
3. **PersonaPlex is genuine full-duplex** — 17-stream temporal transformer, not simulated turn-taking
4. **DeltaNet hybrid attention** in Qwen3Chat is cutting-edge — linear O(n) attention interleaved with full attention
5. **Streaming audio pipeline** is production-grade with cross-fade, pre-buffering, silence compression
6. **OpenAI Realtime API wire compatibility** makes the WebSocket server a local drop-in replacement
7. **Automated distribution** from GitHub Release → binary + metallib → Homebrew formula update
8. **Encoder FP16 / Decoder quantized** pattern is consistent across all MLX models — accuracy where it matters, memory savings where it doesn't
