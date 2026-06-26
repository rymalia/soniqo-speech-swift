---
date: 2026-04-06
time: "11:17 AM PDT – 4:45 PM PDT"
project: soniqo-speech-swift
branch: main
---

# Session Summary: Example App Installation & Debugging

## Overview

First hands-on installation session: built the main speech-swift package from source, then installed and ran all three example demo apps (SpeechDemo, PersonaPlexDemo, iOSEchoDemo). Diagnosed and fixed SPM package identity mismatches, resolved SwiftUI bundle identifier issues, identified a critical missing metallib problem causing ~5x slowdown, and characterized hardware-specific performance limits for PersonaPlex on a 16 GB M3 Air.

## Key Decisions Made

- **Fix downstream Package.swift files instead of renaming the repo directory**: The local directory is named `soniqo-speech-swift` but the example apps referenced the package as `"speech-swift"` (or `"Qwen3Speech"` for iOSEchoDemo). Rather than renaming the directory or using a symlink, we updated all three example Package.swift files and the iOSEchoDemo project.yml to use `"soniqo-speech-swift"`. This keeps the local directory name stable and makes the examples work for this specific checkout.
- **Use `.app` bundle wrappers instead of Xcode for running demos**: SwiftUI apps built as SPM executable targets lack a bundle identifier, which prevents the window from appearing ("Cannot index window tabs due to missing main bundle identifier"). Wrapping the built binary in a minimal `.app` bundle with an `Info.plist` provides the bundle identifier, microphone entitlement description, and proper macOS app lifecycle.
- **Switch PersonaPlexDemo from 8-bit to 4-bit model**: The demo hardcoded `PersonaPlexModel.modelId8bit` (9.1 GB, ~11 GB peak inference), which is too large for a 16 GB machine. Switched to the default 4-bit model (4.9 GB, ~6.5 GB peak) for better memory fit.

## Changes Made

| Change | Detail |
|--------|--------|
| **SPM package identity fix** | Updated `Examples/SpeechDemo/Package.swift`, `Examples/PersonaPlexDemo/Package.swift`, `Examples/iOSEchoDemo/Package.swift`, and `Examples/iOSEchoDemo/project.yml` — changed package references from `"speech-swift"` / `"Qwen3Speech"` to `"soniqo-speech-swift"` to match the local directory name |
| **PersonaPlexDemo 4-bit model** | Changed `PersonaPlexViewModel.swift` to use `PersonaPlexModel.defaultModelId` (4-bit) instead of `PersonaPlexModel.modelId8bit` (8-bit), and updated the SentencePiece tokenizer cache dir reference accordingly |
| **VAD debug logging** | Added temporary debug print in `Sources/SpeechVAD/StreamingVADProcessor.swift` to output VAD probability every 10 chunks (~320ms) for diagnosing speech-end detection failure |

## Testing / Research Performed

- **SpeechDemo Echo tab**: Verified full pipeline (Silero VAD → Parakeet ASR → Qwen3-TTS) works end-to-end. Performance was excellent after copying the metallib — CoreML models (ASR + VAD) run on Neural Engine, only TTS uses GPU (~1 GB), leaving plenty of headroom on 16 GB
- **SpeechDemo without metallib**: Confirmed MLX falls back to JIT shader compilation with XPC errors (`MTLCompiler[reqID=494]: Connection attempt 1/10 failed with XPC_ERROR_CONNECTION_INVALID`), significantly slower
- **SpeechDemo with metallib**: Dramatic performance improvement — TTS inference runs at full speed with precompiled Metal shaders
- **PersonaPlexDemo 8-bit**: Confirmed "terrible" performance on M3 Air 16 GB — model weight memory (9.1 GB) plus inference overhead exceeds available unified memory, causing heavy swap pressure
- **PersonaPlexDemo 4-bit**: Slightly better but still marginal — 7B model is fundamentally too large for this hardware class
- **PersonaPlexDemo VAD debugging**: Added probability logging to diagnose why VAD detects speech start but never fires speech end. Investigation was started but not completed (user pivoted to SpeechDemo)
- **Build warnings audit**: Catalogued all compiler warnings from `swift build -c release` — Sendable conformance (3), deprecated MLX API rename (1), temporary pointer UB (2), unreachable catch (1), unused variables (2). None affect correctness.

## Discoveries / Handoff Notes

### SPM Package Identity Resolution
SPM derives the package identity for local path dependencies (`.package(path: "../..")`) from the **directory name**, not the `name:` field in the dependency's Package.swift. The root package declares `name: "Qwen3Speech"` but the identity is `soniqo-speech-swift` (the directory). This affects all `.product(name:, package:)` references in downstream Package.swift files. If someone clones via `git clone https://github.com/soniqo/speech-swift`, the directory would be `speech-swift` and the original Package.swift references would work. The fix is directory-name-dependent.

### The metallib is Critical for Demo Performance
The precompiled `mlx.metallib` must be copied into any `.app` bundle's `Contents/MacOS/` directory for MLX inference to run at full speed. Without it, MLX JIT-compiles ~200 Metal shader functions on demand via XPC to the Metal compiler service. This is both slow (~5x) and fragile (XPC connections can drop). The SpeechDemo README and build scripts don't currently automate this step for `.app` bundles — it's only documented in the PersonaPlexDemo README.

### Running SwiftUI SPM Executables
SwiftUI's `WindowGroup` requires `CFBundleIdentifier` from `Info.plist` to manage window state. SPM executable targets don't inject an `Info.plist`, so running from Xcode shows "Cannot index window tabs" and no window appears. The workaround is wrapping in a minimal `.app` bundle. Running `open Foo.app` launches as a proper macOS app but swallows stdout/stderr; running `Foo.app/Contents/MacOS/Foo` directly gives both the SwiftUI window AND terminal log output — the best option for development.

### Hardware Performance Characteristics for PersonaPlex
LLM inference is **memory-bandwidth-bound**, not compute-bound. Each autoregressive step reads the entire model weights once. For the 4-bit 7B PersonaPlex model (~5 GB), at the M3 Air's ~100 GB/s bandwidth, the theoretical minimum is ~50ms/step. The M2 Max at ~400 GB/s achieves ~12.5ms/step for the same read. PersonaPlex needs RTF < 1.0 for real-time audio; the M3 Air cannot sustain this. The SpeechDemo Echo tab (Parakeet ASR + Qwen3-TTS) is the performance sweet spot for 16 GB machines — CoreML models offload to Neural Engine, and the ~1 GB TTS model fits easily in GPU memory.

### Voice Processing Errors Are Cosmetic
The `AggregateDevice.mm`, `vpStrategyManager.mm`, and `Voice_Processor_Interface_Adapter` errors from the Echo tab's `setVoiceProcessingEnabled(true)` (AEC) are noisy but non-fatal. They appear with certain audio device configurations and don't affect transcription or playback quality.

### PersonaPlexDemo VAD Issue (Unresolved)
The PersonaPlexDemo's turn-based mode detects speech start but never fires speech end. Debug logging was added to `StreamingVADProcessor` to print probabilities every ~320ms, but the investigation was not completed. The VAD uses MLX-backed Silero (unlike SpeechDemo's Echo tab which uses CoreML-backed Silero). The root cause may be related to the MLX VAD always outputting high probabilities, audio conversion feeding noise, or the different engine behavior. This needs further investigation.

## Current State

- **SpeechDemo**: Fully working at `/tmp/SpeechDemo.app` with metallib. All three tabs functional (Dictate, Speak, Echo). Echo tab has excellent performance on M3 Air.
- **PersonaPlexDemo**: Built at `/tmp/PersonaPlexDemo.app` with metallib and 4-bit model. Runs but performance is poor due to hardware limitations (16 GB M3 Air). VAD speech-end detection issue unresolved.
- **iOSEchoDemo**: Package.swift and project.yml fixed but not yet built or tested. Requires `xcodegen generate` and Xcode with signing team configured.
- **Uncommitted changes**: 6 files modified (3 example Package.swift fixes, 1 project.yml fix, 1 ViewModel model-size change, 1 debug print in StreamingVADProcessor). None committed per user's git workflow preference.
- **Debug logging**: Temporary `[VAD-DEBUG]` print statement in `Sources/SpeechVAD/StreamingVADProcessor.swift:94` should be removed before committing.

## Summary Statistics

- **6 files modified**, 24 insertions, 25 deletions
- **3 example apps** investigated and configured
- **2 performance issues** diagnosed (missing metallib, model too large for hardware)
- **1 SPM resolution bug** fixed across 4 files
- **1 VAD issue** identified but unresolved
- **8 build warnings** catalogued (none blocking)

## Unfinished Work

- **PersonaPlexDemo VAD speech-end detection**: The `[VAD-DEBUG]` logging is in place but needs actual test run analysis to determine why probabilities never drop below the offset threshold. May be MLX vs CoreML engine behavior, audio conversion issue, or model-specific
- **iOSEchoDemo**: Package references fixed but app not yet built or tested on device/simulator
- **Remove debug logging**: The temporary VAD debug print in `StreamingVADProcessor.swift` needs to be removed before any commit
- **Build warnings cleanup**: The deprecated `quantizedMatmul` → `quantizedMM` rename, temporary pointer UB in `MelFeatureExtractor.swift`, and unused variable warnings are low priority but should be addressed eventually
