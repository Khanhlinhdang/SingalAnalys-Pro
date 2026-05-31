"""
Output adapters for converting decoded RF payloads into human-usable artifacts.

This module provides lightweight adapters for:
- text extraction from decoded bitstreams
- mail-like frame extraction from textual payloads
- audio metadata extraction from demodulated audio streams
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

import numpy as np


@dataclass
class OutputArtifact:
    """A normalized output artifact produced by an adapter."""

    artifact_type: str
    confidence: float
    payload: Dict[str, Any]


OUTPUT_CONTRACT_VERSION = "1.0"


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack uint8/bool bits into bytes."""
    if bits is None or len(bits) == 0:
        return b""
    arr = np.asarray(bits).astype(np.uint8)
    arr = np.clip(arr, 0, 1)

    pad_len = (-len(arr)) % 8
    if pad_len:
        arr = np.concatenate([arr, np.zeros(pad_len, dtype=np.uint8)])

    packed = np.packbits(arr)
    return packed.tobytes()


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    return printable / max(1, len(text))


class TextOutputAdapter:
    """Extract plain-text artifact candidates from decoded bytes."""

    def extract(self, payload_bytes: bytes) -> Optional[OutputArtifact]:
        if not payload_bytes:
            return None

        text = payload_bytes.decode("utf-8", errors="ignore")
        text = text.strip("\x00")
        if len(text) < 8:
            return None

        ratio = _printable_ratio(text)
        if ratio < 0.70:
            return None

        preview = text[:240]
        confidence = min(0.99, 0.5 + 0.5 * ratio)
        return OutputArtifact(
            artifact_type="text",
            confidence=confidence,
            payload={
                "length": len(text),
                "preview": preview,
                "printable_ratio": ratio,
            },
        )


class MailOutputAdapter:
    """Extract RFC822-like mail/message artifact candidates from text."""

    REQUIRED_HEADERS = ("from:", "to:", "subject:")

    def extract(self, text: str) -> Optional[OutputArtifact]:
        if not text:
            return None

        low = text.lower()
        hits = sum(h in low for h in self.REQUIRED_HEADERS)
        if hits < 2:
            return None

        header_candidates = re.findall(
            r"(?im)^(from|to|subject|date|message-id)\s*:\s*(.+)$",
            text,
        )
        if not header_candidates:
            return None

        headers: Dict[str, str] = {}
        for key, value in header_candidates:
            k = key.lower()
            if k not in headers:
                headers[k] = value.strip()[:200]

        confidence = min(0.98, 0.4 + 0.15 * len(headers) + 0.15 * hits)
        return OutputArtifact(
            artifact_type="mail",
            confidence=confidence,
            payload={
                "headers": headers,
                "summary": {
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "subject": headers.get("subject", ""),
                },
            },
        )


class AudioOutputAdapter:
    """Extract audio metadata from demodulated audio arrays."""

    def extract(self, demodulated_audio: np.ndarray, sample_rate: float) -> Optional[OutputArtifact]:
        if demodulated_audio is None or len(demodulated_audio) == 0:
            return None

        audio = np.asarray(demodulated_audio)
        if not np.issubdtype(audio.dtype, np.number):
            return None

        rms = float(np.sqrt(np.mean(np.square(np.abs(audio)))))
        peak = float(np.max(np.abs(audio)))
        duration_sec = float(len(audio) / max(1.0, sample_rate))

        # Heuristic confidence: non-silent and bounded signal.
        confidence = 0.2
        if rms > 1e-4:
            confidence += 0.4
        if peak > rms:
            confidence += 0.2
        if duration_sec >= 0.05:
            confidence += 0.2
        confidence = float(min(0.95, confidence))

        return OutputArtifact(
            artifact_type="audio",
            confidence=confidence,
            payload={
                "sample_rate": float(sample_rate),
                "samples": int(len(audio)),
                "duration_sec": duration_sec,
                "rms": rms,
                "peak": peak,
            },
        )


class PcmOutputAdapter:
    """Extract PCM-ready payload contract from demodulated audio."""

    def __init__(self, max_samples: int = 200000):
        self.max_samples = max(1024, int(max_samples))

    def extract(self, demodulated_audio: np.ndarray, sample_rate: float) -> Optional[OutputArtifact]:
        if demodulated_audio is None or len(demodulated_audio) == 0:
            return None

        audio = np.asarray(demodulated_audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return None

        max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if max_abs <= 1e-9:
            return None

        normalized = np.clip(audio / max_abs, -1.0, 1.0)
        truncated = False
        if normalized.size > self.max_samples:
            normalized = normalized[: self.max_samples]
            truncated = True

        pcm_s16 = np.round(normalized * 32767.0).astype(np.int16)
        confidence = 0.75 if not truncated else 0.65

        return OutputArtifact(
            artifact_type="pcm",
            confidence=float(confidence),
            payload={
                "sample_rate": float(sample_rate),
                "encoding": "pcm_s16le",
                "channels": 1,
                "sample_count": int(pcm_s16.size),
                "truncated": bool(truncated),
                "samples": pcm_s16.tolist(),
            },
        )


def normalize_artifact_contracts(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize artifact payloads to stable image/video/audio contracts."""
    normalized: List[Dict[str, Any]] = []
    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue

        kind = str(artifact.get("type", "")).lower()
        payload = artifact.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}

        payload.setdefault("contract_version", OUTPUT_CONTRACT_VERSION)
        payload.setdefault("artifact_kind", kind)

        if kind == "image":
            payload.setdefault("media_class", "image")
            payload.setdefault("format", payload.get("format", "grayscale_8bit"))
            summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
            if "width" in summary:
                payload.setdefault("width", summary.get("width"))
            if "height" in summary:
                payload.setdefault("height", summary.get("height"))
        elif kind == "video":
            payload.setdefault("media_class", "video")
            payload.setdefault("container", payload.get("container", "raw"))
            payload.setdefault("codec", payload.get("codec", "unknown"))
            payload.setdefault("frame_count", payload.get("frame_count", 0))
            payload.setdefault("fps", payload.get("fps", None))
        elif kind in {"audio", "pcm"}:
            payload.setdefault("media_class", "audio")
            payload.setdefault("sample_rate", payload.get("sample_rate", None))

        normalized.append(
            {
                "type": artifact.get("type"),
                "confidence": float(artifact.get("confidence", 0.0)),
                "payload": payload,
            }
        )

    return normalized


def extract_all_artifacts(
    decoded_bits: Optional[np.ndarray],
    demodulated_audio: Optional[np.ndarray],
    sample_rate: float,
) -> List[Dict[str, Any]]:
    """Run all adapters and return normalized artifact dictionaries."""
    artifacts: List[OutputArtifact] = []

    payload_bytes = bits_to_bytes(decoded_bits) if decoded_bits is not None else b""

    text_adapter = TextOutputAdapter()
    mail_adapter = MailOutputAdapter()
    audio_adapter = AudioOutputAdapter()
    pcm_adapter = PcmOutputAdapter()

    text_art = text_adapter.extract(payload_bytes)
    if text_art:
        artifacts.append(text_art)
        mail_art = mail_adapter.extract(text_art.payload.get("preview", ""))
        if mail_art:
            artifacts.append(mail_art)

    audio_art = audio_adapter.extract(demodulated_audio, sample_rate)
    if audio_art:
        artifacts.append(audio_art)

    pcm_art = pcm_adapter.extract(demodulated_audio, sample_rate)
    if pcm_art:
        artifacts.append(pcm_art)

    raw = [
        {
            "type": art.artifact_type,
            "confidence": art.confidence,
            "payload": art.payload,
        }
        for art in artifacts
    ]
    return normalize_artifact_contracts(raw)
