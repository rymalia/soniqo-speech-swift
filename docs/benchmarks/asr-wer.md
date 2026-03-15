# ASR Word Error Rate (WER) Benchmark

## Datasets

- **LibriSpeech test-clean** — 2620 utterances, ~5.4 hours, English read speech (standard ASR benchmark)
- **FLEURS** — multilingual (10 languages), ~400-900 utterances per language, freely downloadable
- **CommonVoice** — multilingual (EN, ZH, DE, ES, FR), crowd-sourced read speech (requires manual download from commonvoice.mozilla.org)

## Results

| Model | Engine | Bits | Size | WER% | RTF | Model Load | Warmup |
|-------|--------|------|------|------|-----|------------|--------|
| Parakeet TDT 0.6B | CoreML (ANE) | INT8 | 634 MB | 2.74 | 0.089 | 128.9s | 2.0s |
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 960 MB | 2.80 | 0.025 | 2.4s | 0.3s |
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 675 MB | 3.34 | 0.023 | 2.4s | 0.3s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT4 | 332 MB | 3.68 | 0.295 | 23.3s | 2.4s |
| Qwen3-ASR 0.6B | CoreML+MLX | 8-bit | 960 MB | — | 0.026 | 2.5s | 0.4s |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

**Key observations:**
- Parakeet INT8 achieves the best WER (2.74%) but has a slow cold start (128.9s CoreML compilation)
- Qwen3-ASR MLX is 10x faster to load (2.4s vs 23-129s) and has the fastest RTF (0.023)
- CoreML+MLX hybrid uses ANE for encoder + GPU for decoder, freeing GPU for other tasks
- Parakeet INT8 is 3.3x faster than INT4 (RTF 0.089 vs 0.298, verified over 3 runs). CoreML's Neural Engine processes INT8 natively; INT4 palettization adds per-operation dequantization overhead that outweighs the smaller model size
- CoreML cold start (first-ever load) compiles a device-specific execution plan: 129s for INT8, 23s for INT4. Warm start (cached) is 5.4s — CoreML caches compiled plans in `~/Library/Caches/com.apple.CoreML/`. The 129s only happens once per device. Encoder currently uses `.all` compute units; switching to `.cpuAndNeuralEngine` would skip GPU plan compilation

## Comparison with published models

| Model | Params | Size | Precision | WER% (test-clean) | Source |
|-------|--------|------|-----------|-------------------|--------|
| Whisper Large v3 Turbo | 809M | 1.6 GB | FP16 | 2.5 | OpenAI (2024) |
| Whisper Large v3 | 1.5B | 3.1 GB | FP16 | 2.7 | OpenAI (2023) |
| **Parakeet TDT 0.6B INT8** | **600M** | **634 MB** | **INT8** | **2.74** | **This benchmark** |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **960 MB** | **8-bit** | **2.80** | **This benchmark** |
| Whisper Medium | 769M | 1.5 GB | FP16 | 3.0 | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **675 MB** | **4-bit** | **3.34** | **This benchmark** |
| Whisper Small | 244M | 483 MB | FP16 | 3.4 | OpenAI (2022) |
| **Parakeet TDT 0.6B INT4** | **600M** | **332 MB** | **INT4** | **3.68** | **This benchmark** |
| FireRedASR2-AED | 1B | ~2 GB | FP16 | 4.57 | Xiaohongshu (2025) |
| Whisper Base | 74M | 142 MB | FP16 | 5.0 | OpenAI (2022) |

Whisper numbers from original papers (FP16 inference).

## Multilingual results (FLEURS)

Qwen3-ASR 0.6B 4-bit on FLEURS test splits. CER used for CJK languages (no word boundaries).

| Language | Code | Metric | Error% | Utterances |
|----------|------|--------|--------|-----------|
| Spanish | es_419 | WER | 6.44 | 908 |
| English | en_us | WER | 6.57 | 647 |
| Chinese | cmn_hans_cn | CER | 8.41 | 945 |
| German | de_de | WER | 9.45 | 862 |
| French | fr_fr | WER | 11.42 | 676 |
| Japanese | ja_jp | CER | 16.11 | 650 |
| Russian | ru_ru | WER | 16.35 | 775 |
| Korean | ko_kr | WER | 19.95 | 382 |
| Hindi | hi_in | WER | 25.93 | 418 |
| Arabic | ar_eg | WER | 33.47 | 428 |

Best on European languages (EN/ES/DE/FR under 12%). CJK is reasonable (ZH 8.4%, JA 16.1%). Arabic and Hindi are weaker — likely less training data for these languages in the 0.6B model.

## Compression delta

How much accuracy do we lose by quantizing to lower bit widths? This establishes the baseline quality cost of our current quantization before trying more advanced techniques like mixed-bit allocation or outlier decomposition.

| Variant | WER% | Substitutions | Insertions | Deletions | Total errors | Size |
|---------|------|---------------|------------|-----------|-------------|------|
| Qwen3 0.6B 8-bit | 2.80 | 1111 | 92 | 268 | 1471 | 960 MB |
| Qwen3 0.6B 4-bit | 3.34 | 1323 | 123 | 308 | 1754 | 675 MB |
| Delta | +0.54 | +212 | +31 | +40 | +283 | -30% |
| Parakeet TDT INT8 | 2.74 | 990 | 125 | 308 | 1423 | 634 MB |
| Parakeet TDT INT4 | 3.68 | 1279 | 138 | 493 | 1910 | 332 MB |
| Delta | +0.94 | +289 | +13 | +185 | +487 | -48% |

**Qwen3-ASR**: 4-bit adds 0.54% WER (19% more errors) for 30% size reduction.

**Parakeet TDT**: INT4 adds 0.94% WER (34% more errors) for 48% size reduction. INT4 loses more quality than Qwen3's 4-bit — CoreML k-means palettization is less precise than MLX group quantization at low bit widths.

## Reproduction

```bash
make build
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B-8bit
python scripts/benchmark_asr.py --batch --engine parakeet
python scripts/benchmark_asr.py --batch --engine parakeet --model int8
```

First run downloads LibriSpeech test-clean (~350 MB). Results saved to `benchmarks/librispeech/`.

### FLEURS (multilingual, auto-download)

```bash
python scripts/benchmark_asr.py --dataset fleurs --language en_us --batch
python scripts/benchmark_asr.py --dataset fleurs --language cmn_hans_cn --batch
python scripts/benchmark_asr.py --dataset fleurs --language de_de --batch
```
