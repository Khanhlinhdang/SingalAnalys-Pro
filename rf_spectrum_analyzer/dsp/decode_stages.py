"""Decode depth stages for deinterleave and descramble processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DecodeStageResult:
    """Result container for decode depth stage execution."""

    output_bits: np.ndarray
    operations_applied: List[str]
    deinterleave_applied: bool
    descramble_applied: bool
    confidence: float

    def to_metrics(self, input_len: int) -> Dict[str, float]:
        output_len = int(len(self.output_bits))
        ratio = 0.0
        if input_len > 0:
            ratio = float(abs(output_len - input_len) / input_len)

        return {
            "input_bits": int(input_len),
            "output_bits": output_len,
            "length_delta_ratio": ratio,
            "deinterleave_applied": bool(self.deinterleave_applied),
            "descramble_applied": bool(self.descramble_applied),
            "operations_count": int(len(self.operations_applied)),
            "confidence": float(self.confidence),
        }


class DecodeDepthPipeline:
    """Standardized stage pipeline for bit-level post-demod processing."""

    def process(
        self,
        bits: Optional[np.ndarray],
        modulation_type: Optional[str] = None,
        protocol_hint: Optional[str] = None,
    ) -> DecodeStageResult:
        norm_bits = self._normalize_bits(bits)
        operations: List[str] = ["normalize_bits"]

        deinterleaved, deinterleave_applied = self._apply_deinterleave(norm_bits, modulation_type)
        if deinterleave_applied:
            operations.append("block_deinterleave")

        descrambled, descramble_applied = self._apply_descramble(deinterleaved, protocol_hint)
        if descramble_applied:
            operations.append("descramble_lfsr")

        # Conservative confidence until protocol-specific chains are integrated.
        confidence = 0.35
        if deinterleave_applied:
            confidence += 0.15
        if descramble_applied:
            confidence += 0.2
        confidence = float(min(0.9, confidence))

        return DecodeStageResult(
            output_bits=descrambled,
            operations_applied=operations,
            deinterleave_applied=deinterleave_applied,
            descramble_applied=descramble_applied,
            confidence=confidence,
        )

    def _normalize_bits(self, bits: Optional[np.ndarray]) -> np.ndarray:
        if bits is None:
            return np.array([], dtype=np.uint8)

        arr = np.asarray(bits).flatten()
        if arr.size == 0:
            return np.array([], dtype=np.uint8)

        if not np.issubdtype(arr.dtype, np.integer):
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        return arr & 1

    def _apply_deinterleave(self, bits: np.ndarray, modulation_type: Optional[str]) -> (np.ndarray, bool):
        """Apply conservative block deinterleave for likely interleaved schemes."""
        if bits.size == 0:
            return bits, False

        mod = (modulation_type or "").upper()
        if mod not in {"QPSK", "QAM16", "QAM64"}:
            return bits, False

        # Conservative mode: only deinterleave if exact 2-row matrix is possible and long enough.
        if bits.size < 64 or bits.size % 2 != 0:
            return bits, False

        reshaped = bits.reshape(2, -1)
        deinterleaved = reshaped.T.flatten()
        return deinterleaved.astype(np.uint8), True

    def _apply_descramble(self, bits: np.ndarray, protocol_hint: Optional[str]) -> (np.ndarray, bool):
        """Apply protocol-aware descramble where a stable polynomial is known."""
        if bits.size == 0:
            return bits, False

        hint = (protocol_hint or "").lower()
        if "inmarsat" not in hint:
            return bits, False

        # Inmarsat-C polynomial: x^23 + x^18 + 1
        state = 0x3FFFFF
        out = np.zeros_like(bits, dtype=np.uint8)
        for i, bit in enumerate(bits):
            lfsr_bit = (state >> 22) & 1
            out[i] = bit ^ lfsr_bit
            feedback = ((state >> 22) ^ (state >> 17)) & 1
            state = ((state << 1) | feedback) & 0x7FFFFF

        return out, True


def create_default_decode_depth_pipeline() -> DecodeDepthPipeline:
    return DecodeDepthPipeline()
