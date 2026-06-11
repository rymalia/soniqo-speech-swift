# Nemotron-3.5 ASR Streaming — architecture

Multilingual streaming ASR from NVIDIA, ported to CoreML for Apple Silicon via the `NemotronStreamingASR` target. 600 M parameters, 40 language-locales, native punctuation and capitalization. Cache-aware FastConformer encoder with a prompt-conditioned RNN-T decoder.

## Source

- Upstream: [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (gated)
- Reference architecture: NeMo `EncDecRNNTBPEModelWithPrompt`
- Conversion pipeline: `speech-models/models/nemotron-asr-streaming-multilingual/export/`

## Top-level pipeline

```
raw 16 kHz audio
  → mel preprocessor (NeMo FilterbankFeatures-equivalent, 128 bins, 25 ms / 10 ms)
  → cache-aware Conformer encoder (24 layers, d_model=1024, 580 M params)
  → prompt-kernel language conditioning (one-hot 128-slot → concat → 2-layer MLP)
  → RNN-T greedy decoder (2-layer LSTM, pred_hidden=640, blank id=13087)
  → optional word boosting over joint logits
  → text tokens (vocab 13087, with `<lang-XX>` markers)
```

The encoder produces one frame every 80 ms (8× subsampling from 10 ms mel hop). All four chunk modes share the same encoder and decoder weights; only the attention right-context changes.

## Cache-aware streaming

The encoder runs over short audio chunks while preserving four caches across calls:

| Cache | Shape | Purpose |
|-------|-------|---------|
| `pre_cache` (mel) | `[1, 128, 9]` | last 9 mel frames carried into next chunk's pre-subsampling stack |
| `cache_last_channel` (KV) | `[24, 1, 56, 1024]` | per-layer attention K/V over the 56-frame left context |
| `cache_last_time` (conv) | `[24, 1, 1024, 8]` | per-layer conv left-pad equivalent (kernel=9, K-1=8) |
| `cache_last_channel_len` | `[1]` int32 | number of valid frames currently in KV cache |

The decoder LSTM state (`h`, `c`) is also persisted between chunks; the joint network is stateless.

## Word boosting

`NemotronStreamingASR` implements word boosting decoder-side, between `joint.prediction(...)` and greedy token selection in `RNNTGreedyDecoder`. Boosted phrases are tokenized with the Nemotron SentencePiece tokenizer when `tokenizer.model` is present, stored in a phrase-prefix trie, and applied as token-level logit bias during decoding. This is an RNN-T shallow-fusion path for the current CoreML bundle, not the full NVIDIA CTC-WS algorithm.

The implementation does not allow word boosting to replace the RNN-T blank token. Blank advances decoding to the next encoder frame; letting a boosted phrase beat blank can pin the decoder on one frame and produce repeated-token garbage. Boosting is only applied when the unboosted greedy token is non-blank.

When `ctc.mlmodelc` is present, Speech Swift can run it on each chunk's `encoded_output` and store recent CTC log-prob frames. If word boosting tries to replace the normal greedy token, the matched token path must also have ordered CTC evidence near the current encoder frame. Pure SentencePiece boundary tokens (`▁`) are ignored by this acoustic check.

For this to work, the bundle must include an auxiliary CTC head from a CTC or hybrid RNN-T/CTC checkpoint. The public `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint inspected on June 10, 2026 is RNNT-only: its `model_config.yaml` target is `EncDecRNNTBPEModelWithPrompt`, it has no `aux_ctc` section, and its weights contain no CTC or auxiliary-CTC tensors. Bundles built from that checkpoint report `shallowFusionOnly` via `wordBoostingValidationStatus`.

## Chunk modes (att_context_size from .nemo config)

| chunk_ms | output frames | right ctx | latency target |
|----------|--------------:|----------:|---------------:|
| 80   | 1  | 0  | ultra-low (mel input = 8 frames) |
| 320  | 4  | 3  | default (used in published bundle) |
| 560  | 7  | 6  | balanced |
| 1120 | 14 | 13 | near-offline |

Bundles published as `aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8` use chunk_ms = 320.

## Prompt-kernel language conditioning

Each session receives a one-hot `language_mask` of shape `[1, 128]`. The encoder broadcasts this across every output frame, concatenates with the 1024-dim encoded vector (→ 1152 dims), and projects through a small MLP:

```
encoded (B, T, 1024) ⊕ lang_mask (B, T, 128)
  → Linear(1152 → 2048) → ReLU → Linear(2048 → 1024)
  → final encoded
```

Slot mapping ships in `languages.json` (e.g. `"en-US": 0`, `"de-DE": 9`, `"ja-JP": 10`, `"auto": 101`). 128 slots total: 84 used languages + `auto` + reserved aliases.

## Vocabulary

- 13 087 SentencePiece pieces + 1 blank id (13087)
- Native punctuation tokens (`.`, `,`, `?`, `!`, etc.) as regular vocab entries
- Per-language tag tokens (`<en-US>`, `<de-DE>`, ...) emitted as suffix on each utterance; the Swift wrapper strips these via a regex in WER-style normalization

## CoreML bundle layout

After running `convert.py --chunk-ms 320 --compile`:

```
encoder.mlmodelc/    565 MB  INT8 palettized, ANE + GPU friendly
decoder.mlmodelc/     29 MB  FP16 LSTM, runs on CPU
joint.mlmodelc/       18 MB  FP16 dense, runs on CPU
ctc.mlmodelc/         optional auxiliary CTC verifier for word boosting
config.json           streaming geometry + dims for the loader
vocab.json            id → piece
tokenizer.model       optional; real SentencePiece model for word boosting phrase tokenization
languages.json        promptDictionary: lang tag → slot
```

`.mlmodelc` (compiled) ships rather than `.mlpackage` per the speech-models policy — on-device `MLModel.compileModel()` produces non-deterministic output across simulator vs device runtimes, breaking parity.

### Exporting the CTC verifier

The auxiliary CTC verifier can only be exported from a CTC or hybrid RNN-T/CTC `.nemo` checkpoint whose config contains `aux_ctc`:

```bash
python scripts/export_nemotron_ctc_coreml.py \
  /path/to/hybrid-rnnt-ctc-nemotron.nemo \
  /path/to/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8 \
  --compile
```

The script refuses RNNT-only checkpoints up front. For a compatible hybrid checkpoint, it restores the `.nemo`, reads `model.ctc_decoder`, wraps it to accept the existing CoreML encoder output shape `[1, outputFrames, encoderHidden]`, and writes `ctc.mlmodelc` into the bundle.

## Differences vs older `nemotron-speech-streaming-en-0.6b`

| | English 0.6b (older) | Multilingual 3.5 0.6b |
|---|---|---|
| Vocab | 1 024 + 1 blank | 13 087 + 1 blank |
| Attention left context | 70 frames | 56 frames |
| Languages | English only | 76 |
| Prompt kernel | none | one-hot 128-slot, concat + 2-layer MLP |
| Default chunk | 160 ms | 320 ms |
| .nemo size | ~590 MB | ~2.37 GB |

The Swift `NemotronStreamingConfig` decoder accepts both layouts — older English bundles missing `numPrompts` default to 128, and `attentionContext` accepts either the new `attentionLeftContext` field name or the old `attentionContext` name.

## Conversion equivalence

Round-trip verified Δ-CER vs the fp32 NeMo source on 6 languages × 50 FLEURS samples each (50 + 60 chunks per sample):

| variant | Δ avg CER vs fp32 |
|---------|------------------:|
| CoreML INT8 | +0.13 pp |
| MLX bf16    | +0.20 pp |
| MLX int8    | +0.32 pp |
| MLX int4    | +2.32 pp |

CoreML INT8 / MLX bf16 / MLX int8 are essentially lossless; int4 is meaningfully lossier especially on Hindi and Japanese.

## References

- Stateful Conformer with Cache-based Inference: <https://arxiv.org/pdf/2312.17279>
- Fast Conformer with Linearly Scalable Attention: <https://arxiv.org/abs/2305.05084>
