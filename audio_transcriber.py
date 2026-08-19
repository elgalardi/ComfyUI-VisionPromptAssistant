from __future__ import annotations

import gc
import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
from comfy_api.latest import io


WHISPER_MODELS = ["large-v3", "large-v3-turbo"]
LANGUAGES = [
    "auto", "en", "es", "ja", "zh", "hi", "ar", "fr", "bn", "pt",
    "ru", "id", "de", "it", "ko",
]
DEVICES = ["auto", "cuda", "cpu"]
COMPUTE_TYPES = ["auto", "float16", "int8_float16", "int8", "float32"]


def _mono_16khz(audio: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    if not isinstance(audio, Mapping) or "waveform" not in audio:
        raise ValueError("Local Whisper Transcribe requires a ComfyUI AUDIO value.")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform) or waveform.ndim not in (1, 2, 3):
        raise ValueError("AUDIO waveform must be a 1D, 2D, or 3D tensor.")
    sample_rate = int(audio.get("sample_rate", 0))
    if sample_rate <= 0:
        raise ValueError("AUDIO sample_rate must be positive.")
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        waveform = waveform[0]
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        target = max(1, round(int(waveform.shape[-1]) * 16000 / sample_rate))
        waveform = F.interpolate(
            waveform.unsqueeze(0), size=target, mode="linear", align_corners=False
        ).squeeze(0)
    samples = waveform.squeeze(0).contiguous().numpy()
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return samples, rms


def _release_whisper(model) -> None:
    if model is not None:
        backend = getattr(model, "model", None)
        unload = getattr(backend, "unload_model", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class LocalWhisperTranscribe(io.ComfyNode):
    """Short-lived faster-whisper transcription with explicit VRAM release."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LocalWhisperTranscribe",
            display_name="Local Whisper Transcribe",
            category="Vision Prompt Assistant/Audio",
            search_aliases=[
                "faster whisper", "large v3", "audio transcript", "speech to text"
            ],
            description=(
                "Transcribes ComfyUI AUDIO locally with faster-whisper and releases "
                "the model before MiniMax H3 loads."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Combo.Input("model", options=WHISPER_MODELS, default="large-v3"),
                io.Combo.Input("language", options=LANGUAGES, default="auto"),
                io.Combo.Input("device", options=DEVICES, default="auto"),
                io.Combo.Input(
                    "compute_type", options=COMPUTE_TYPES, default="auto",
                    tooltip=(
                        "auto uses float16 on CUDA and int8 on CPU. int8_float16 "
                        "uses less VRAM with a small possible accuracy tradeoff."
                    ),
                ),
                io.Int.Input("beam_size", default=5, min=1, max=10),
                io.Boolean.Input("vad_filter", default=True),
                io.String.Input(
                    "initial_prompt", default="", multiline=True, optional=True,
                    tooltip="Optional vocabulary or context hint; it is not added to the result.",
                ),
            ],
            outputs=[
                io.String.Output("transcript"),
                io.String.Output("segments_json"),
                io.String.Output("detected_language"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio,
        model: str,
        language: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
        initial_prompt: str = "",
    ) -> io.NodeOutput:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is not installed. Reinstall this node's dependencies "
                "or run: python_embeded\\python.exe -m pip install faster-whisper"
            ) from error

        samples, rms = _mono_16khz(audio)
        if samples.size < 1 or rms < 1e-5:
            status = f"Silent or empty audio; Whisper skipped (RMS {rms:.6f})."
            return io.NodeOutput("", "[]", "", status)

        resolved_device = str(device)
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but PyTorch reports no CUDA device.")
        resolved_compute = str(compute_type)
        if resolved_compute == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"
        cache_dir = os.path.join(folder_paths.models_dir, "faster-whisper")
        os.makedirs(cache_dir, exist_ok=True)

        whisper = None
        try:
            whisper = WhisperModel(
                str(model),
                device=resolved_device,
                compute_type=resolved_compute,
                download_root=cache_dir,
            )
            segments_iter, info = whisper.transcribe(
                samples,
                language=None if language == "auto" else str(language),
                task="transcribe",
                beam_size=int(beam_size),
                vad_filter=bool(vad_filter),
                initial_prompt=str(initial_prompt or "").strip() or None,
                condition_on_previous_text=True,
            )
            records = []
            texts = []
            for segment in segments_iter:
                text = str(segment.text or "").strip()
                if text:
                    texts.append(text)
                records.append({
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                })
            transcript = " ".join(texts).strip()
            detected = str(getattr(info, "language", "") or "")
            probability = float(getattr(info, "language_probability", 0.0) or 0.0)
            duration = samples.size / 16000.0
            status = (
                f"{model} · {resolved_device}/{resolved_compute} · "
                f"{duration:.2f}s · {len(records)} segments · RMS {rms:.6f} · "
                f"language {detected or 'unknown'} ({probability:.1%})"
            )
            return io.NodeOutput(
                transcript,
                json.dumps(records, ensure_ascii=False, indent=2),
                detected,
                status,
            )
        finally:
            _release_whisper(whisper)


__all__ = ["LocalWhisperTranscribe"]
