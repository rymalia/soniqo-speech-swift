#!/usr/bin/env python3
"""
Export the auxiliary Nemotron CTC head to CoreML.

This script is intentionally narrow: it does not re-export the streaming
encoder. It restores the original NVIDIA .nemo checkpoint, takes the model's
auxiliary CTC decoder, wraps it so it accepts Speech Swift's existing
CoreML encoder output shape [B, T, D], and writes ctc.mlpackage or ctc.mlmodelc.

Expected bundle after running with --compile:

  encoder.mlmodelc/
  decoder.mlmodelc/
  joint.mlmodelc/
  ctc.mlmodelc/
  vocab.json
  tokenizer.model
  languages.json
  config.json

Usage:

  python scripts/export_nemotron_ctc_coreml.py \
      /path/to/nemotron-3.5-asr-streaming-0.6b.nemo \
      /path/to/CoreMLBundle \
      --compile

Python deps:

  pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
  pip install "numpy<2" torch coremltools sentencepiece omegaconf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


def read_config(bundle_dir: Path) -> dict:
    config_url = bundle_dir / "config.json"
    if not config_url.exists():
        raise FileNotFoundError(f"Missing config.json at {config_url}")
    return json.loads(config_url.read_text())


def assert_nemo_can_restore_ctc_head(nemo_path: Path) -> None:
    """Fail fast when the checkpoint is RNNT-only.

    NVIDIA's CTC-WS word boosting requires CTC log probabilities. An RNNT-only
    checkpoint cannot produce those log probabilities, even if the encoder is
    shared with a similarly shaped hybrid model.
    """

    with tarfile.open(nemo_path, "r:*") as archive:
        try:
            config_file = archive.extractfile("model_config.yaml")
        except KeyError as error:
            raise RuntimeError(f"{nemo_path} does not contain model_config.yaml") from error

        if config_file is None:
            raise RuntimeError(f"{nemo_path} does not contain readable model_config.yaml")

        config_text = config_file.read().decode("utf-8", errors="replace")

    target_match = re.search(r"^target:\s*(\S+)", config_text, re.MULTILINE)
    target = target_match.group(1) if target_match else "<missing>"

    if "HybridRNNTCTC" not in target and "CTC" not in target:
        raise RuntimeError(
            "This .nemo checkpoint restores as "
            f"{target}, so NeMo loads it as RNNT-only and does not create model.ctc_decoder. "
            "To build ctc.mlmodelc you need a CTC or hybrid RNN-T/CTC Nemotron checkpoint "
            "whose saved target restores a CTC decoder."
        )

    if "aux_ctc:" not in config_text:
        raise RuntimeError(
            "This .nemo checkpoint does not declare an aux_ctc section, so it is RNNT-only "
            "and has no CTC head to export. To build ctc.mlmodelc you need a CTC or "
            "hybrid RNN-T/CTC Nemotron checkpoint whose config contains aux_ctc."
        )


def compile_model(mlpackage: Path, output_dir: Path) -> Path:
    compiled = output_dir / "ctc.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(output_dir)],
        check=True,
    )

    produced = output_dir / f"{mlpackage.stem}.mlmodelc"
    if produced != compiled:
        if compiled.exists():
            shutil.rmtree(compiled)
        produced.rename(compiled)
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Nemotron auxiliary CTC head to CoreML")
    parser.add_argument("nemo_path", type=Path, help="Original NVIDIA .nemo checkpoint")
    parser.add_argument("bundle_dir", type=Path, help="Existing CoreML bundle directory")
    parser.add_argument("--output-name", default="ctc", help="Output model basename")
    parser.add_argument("--compile", action="store_true", help="Compile .mlpackage to .mlmodelc")
    parser.add_argument("--deployment", default="macOS14", help="CoreML deployment target: macOS13/macOS14/iOS17/iOS18")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    assert_nemo_can_restore_ctc_head(args.nemo_path)

    import coremltools as ct
    import nemo.collections.asr as nemo_asr
    import numpy as np
    import torch

    class CTCHeadWrapper(torch.nn.Module):
        """Adapt Speech Swift encoder output [B, T, D] to NeMo CTC input [B, D, T]."""

        def __init__(self, ctc_decoder: torch.nn.Module):
            super().__init__()
            self.ctc_decoder = ctc_decoder

        def forward(self, encoded_output: torch.Tensor) -> torch.Tensor:
            encoded_channel_first = encoded_output.transpose(1, 2)
            return self.ctc_decoder(encoder_output=encoded_channel_first)

    config = read_config(bundle_dir)
    chunk_frames = int(config["streaming"]["outputFrames"])
    hidden = int(config["encoderHidden"])

    model = nemo_asr.models.ASRModel.restore_from(str(args.nemo_path), map_location="cpu")
    model.eval()

    if not hasattr(model, "ctc_decoder"):
        raise RuntimeError("Restored Nemotron model does not expose ctc_decoder")

    wrapper = CTCHeadWrapper(model.ctc_decoder).eval()
    example = torch.randn(1, chunk_frames, hidden, dtype=torch.float32)

    with torch.no_grad():
        ref = wrapper(example)
    print(f"CTC output shape: {tuple(ref.shape)}")

    traced = torch.jit.trace(wrapper, example)

    target = getattr(ct.target, args.deployment)
    mlpackage = bundle_dir / f"{args.output_name}.mlpackage"
    if mlpackage.exists():
        shutil.rmtree(mlpackage)

    coreml_model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        inputs=[
            ct.TensorType(
                name="encoded_output",
                shape=example.shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="ctc_logprobs", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.ALL,
    )
    coreml_model.save(str(mlpackage))
    print(f"Saved {mlpackage}")

    if args.compile:
        compiled = compile_model(mlpackage, bundle_dir)
        print(f"Compiled {compiled}")


if __name__ == "__main__":
    main()
