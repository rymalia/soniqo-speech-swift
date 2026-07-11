# Voice-Cloning TTS Benchmarks

Single-sentence synthesis benchmark for the zero-shot voice-cloning engines,
cloned from the same 12 s reference clip (with its transcript where the engine
accepts one). Roundtrip = Qwen3-ASR transcription of the generated audio
compared against the input text.

Sentence: "The quick brown fox jumps over the lazy dog and rests in the
afternoon sun."

## Results

| Engine | Params | Peak RSS | Audio | Synth | RTF | Roundtrip |
|---|---|---:|---:|---:|---:|---|
| Higgs TTS 3 (clone) | 4B bf16 | 8.6 GB | 6.04 s | 4.73 s | **0.78** | exact |
| Higgs TTS 3 (no reference) | 4B bf16 | 8.3 GB | 4.80 s | 3.77 s | **0.78** | exact |
| F5-TTS (clone, 16 steps, default) | 336M fp16 | 0.8 GB | 5.09 s | 2.91 s | **0.57** | 1-word sub |
| F5-TTS (clone, 32 steps) | 336M fp16 | 0.8 GB | 5.09 s | 5.75 s | 1.13 | 1-word sub |
| F5-TTS (clone, 12 steps) | 336M fp16 | 0.8 GB | 5.09 s | 2.19 s | 0.43 | 1-word sub |
| IndexTTS2 (clone) | 1.5B-class fp16 | 2.8 GB | 5.27 s | ~45 s * | ~9 * | exact |

**Machine**: Apple M5 Pro, 48 GB, release build with compiled metallib.
Synth time excludes model load (Higgs/F5 report it directly); RSS from
`/usr/bin/time -l`, includes weights.

\* IndexTTS2's CLI does not report synthesis-only timing; the figure is wall
clock minus an estimated ~10 s load of its expanded multi-stage bundle
(GPT + S2Mel + BigVGAN + w2v-BERT/MaskGCT/CAMPPlus auxiliaries).

## Notes

- **Higgs RTF includes the decode-loop pipelining** landed alongside this
  benchmark (previously 1.04 clone / 0.82 plain): sampling stays on-device
  with the delay ramp masked on GPU, and `asyncEval` overlaps the next
  forward pass with the CPU-side EOC state machine. Cloning now costs the
  same RTF as plain synthesis; the remaining decode cost is the bf16
  memory-bandwidth roofline of the 4B backbone (~32 frames/s at 25 fps).
- **Higgs reference encoding** (12 s clip → codes) adds ~0.6 s once per
  voice; `encodeReference` returns reusable codes for repeated generations.
- **F5's flow-step count is the whole speed lever** (CFG is already batched):
  the ASR roundtrip is word-identical at 32/16/12 steps and a listening A/B
  found them indistinguishable, so the default moved from 32 to **16**
  (`--f5-steps 32` remains for maximum fidelity). Steps run over the full
  reference+target sequence, so RTF also improves on longer utterances.
- **IndexTTS2** is the heavyweight path (multi-stage GPT + diffusion +
  vocoder); prefer it for its emotion-control surface rather than latency.
- Cloned-voice quality gates for Higgs and F5 (ASR roundtrips en/zh, plus
  es/de/ja for the Python reference) live in the E2E suites; see
  `docs/models/higgs-tts.md` and `docs/models/f5-tts.md`.
