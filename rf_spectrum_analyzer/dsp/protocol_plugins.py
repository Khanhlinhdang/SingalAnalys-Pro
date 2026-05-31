"""Protocol plugin system for post-demodulation payload parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from rf_spectrum_analyzer.dsp.meteor_lrpt_chain import MeteorLrptDecodeChain


def _as_bits(bits: Optional[np.ndarray]) -> np.ndarray:
    if bits is None:
        return np.array([], dtype=np.uint8)
    arr = np.asarray(bits).astype(np.uint8).flatten()
    if arr.size == 0:
        return np.array([], dtype=np.uint8)
    return arr & 1


def _bits_to_text(bits: np.ndarray) -> str:
    if bits.size < 8:
        return ""

    byte_count = int(bits.size // 8)
    if byte_count <= 0:
        return ""

    packed = np.packbits(bits[: byte_count * 8])
    raw = packed.tobytes().decode("ascii", errors="ignore")
    text = "".join(ch for ch in raw if ch.isprintable() or ch in "\n\r\t")
    # Require at least 4 consecutive printable characters to suppress noise artifacts
    import re
    if not re.search(r'[ -~]{4,}', text):
        return ""
    return text


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size < 8:
        return b""

    byte_count = int(bits.size // 8)
    if byte_count <= 0:
        return b""

    packed = np.packbits(bits[: byte_count * 8])
    return packed.tobytes()


def _audio_to_bits(audio: Optional[np.ndarray]) -> np.ndarray:
    if audio is None:
        return np.array([], dtype=np.uint8)

    arr = np.asarray(audio)
    if arr.size == 0:
        return np.array([], dtype=np.uint8)

    if np.iscomplexobj(arr):
        arr = np.real(arr)

    arr = arr.astype(np.float32).flatten()
    if arr.size == 0:
        return np.array([], dtype=np.uint8)

    arr = arr - float(np.mean(arr))
    scale = float(np.std(arr))
    if scale > 1e-9:
        arr = arr / scale

    bits = (arr > 0).astype(np.uint8)
    if bits.size >= 32:
        smooth = np.convolve(bits.astype(np.float32), np.ones(5, dtype=np.float32) / 5.0, mode="same")
        bits = (smooth >= 0.5).astype(np.uint8)
    return bits & 1


@dataclass
class ProtocolDecodeRequest:
    bits: Optional[np.ndarray]
    modulation_type: Optional[str]
    sample_rate: float
    center_freq: Optional[float]
    auxiliary_signal: Optional[np.ndarray] = None


class ProtocolPlugin:
    """Base class for protocol parsers over demodulated bitstreams."""

    plugin_id: str = "base"
    protocol_name: str = "Unknown"

    def can_handle(self, request: ProtocolDecodeRequest) -> float:
        raise NotImplementedError

    def decode(self, request: ProtocolDecodeRequest) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_native_counter_summary(
        self,
        results: List[Dict[str, Any]],
        frame_total: int,
        frame_locks: int,
        average_ber: Optional[float],
        counter_source: str = "plugin_internal",
    ) -> Dict[str, Any]:
        crc_items = [item.get("crc_ok") for item in results if item.get("crc_ok") is not None]
        crc_ok_rate = None
        if crc_items:
            crc_ok_rate = float(sum(1 for item in crc_items if bool(item)) / len(crc_items))

        frame_lock_ratio = None
        per = None
        if frame_total > 0:
            frame_lock_ratio = float(frame_locks / frame_total)
            failed_packets = sum(
                1
                for item in results
                if item.get("frame_locked") is False or item.get("crc_ok") is False
            )
            if failed_packets == 0 and frame_total > frame_locks:
                failed_packets = frame_total - frame_locks
            per = float(failed_packets / frame_total)

        return {
            "ber": average_ber,
            "per": per,
            "crc_ok_rate": crc_ok_rate,
            "crc_available": bool(crc_items),
            "frame_lock_ratio": frame_lock_ratio,
            "frame_locks": int(frame_locks),
            "frame_total": int(frame_total),
            "counter_source": counter_source,
        }


class ProtocolPluginRegistry:
    """Select and execute the best protocol plugin for given demodulated bits."""

    def __init__(self):
        self._plugins: List[ProtocolPlugin] = []

    def register(self, plugin: ProtocolPlugin) -> None:
        self._plugins.append(plugin)

    def decode(
        self,
        bits: Optional[np.ndarray],
        modulation_type: Optional[str],
        sample_rate: float,
        center_freq: Optional[float] = None,
        auxiliary_signal: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        request = ProtocolDecodeRequest(
            bits=bits,
            modulation_type=modulation_type,
            sample_rate=sample_rate,
            center_freq=center_freq,
            auxiliary_signal=auxiliary_signal,
        )

        candidates: List[Dict[str, Any]] = []
        for plugin in self._plugins:
            score = 0.0
            try:
                score = float(plugin.can_handle(request))
            except Exception:
                score = 0.0
            candidates.append({"plugin_id": plugin.plugin_id, "score": max(0.0, min(1.0, score))})

        if not candidates:
            return {
                "matched_protocol": None,
                "confidence": 0.0,
                "results": [],
                "artifacts": [],
                "candidates": [],
            }

        best = max(candidates, key=lambda x: x["score"])
        if best["score"] < 0.25:
            return {
                "matched_protocol": None,
                "confidence": float(best["score"]),
                "results": [],
                "artifacts": [],
                "candidates": candidates,
            }

        plugin = next((p for p in self._plugins if p.plugin_id == best["plugin_id"]), None)
        if plugin is None:
            return {
                "matched_protocol": None,
                "confidence": 0.0,
                "results": [],
                "artifacts": [],
                "candidates": candidates,
            }

        try:
            decode_result = plugin.decode(request)
            decode_result.setdefault("matched_protocol", plugin.plugin_id)
            decode_result.setdefault("confidence", best["score"])
            decode_result.setdefault("results", [])
            decode_result.setdefault("artifacts", [])
            decode_result["candidates"] = candidates
            return decode_result
        except Exception as exc:
            return {
                "matched_protocol": None,
                "confidence": 0.0,
                "results": [],
                "artifacts": [],
                "candidates": candidates,
                "error": str(exc),
            }


class InmarsatProtocolPlugin(ProtocolPlugin):
    """Minimal Inmarsat-C frame extractor using sync + fixed frame model."""

    plugin_id = "inmarsat_c"
    protocol_name = "Inmarsat-C"

    BAND_START_HZ = 1.52e9
    BAND_END_HZ = 1.55e9

    FRAME_LENGTH = 640
    SYNC_WORD = np.array(
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        dtype=np.uint8,
    )
    PACKET_TYPES = {
        0x04: "EGC SafetyNET",
        0x08: "EGC FleetNET",
        0x0C: "EGC System",
        0x14: "Two-Way Message",
        0x1C: "NCS Common Channel",
    }
    FRAME_PROFILES = (
        # Strict: only used when signal quality is good
        {"name": "nominal",           "preamble_bits":  0, "sync_errors": 2, "ber_tolerance_percent": 45, "max_frames": 8},
        {"name": "short_preamble",    "preamble_bits":  8, "sync_errors": 2, "ber_tolerance_percent": 50, "max_frames": 8},
        # Relaxed sync tolerance (Inmarsat-D / IsatM2M typically needs 5-8 error tolerance)
        {"name": "relaxed_sync",      "preamble_bits":  0, "sync_errors": 5, "ber_tolerance_percent": 50, "max_frames": 6},
        {"name": "relaxed_preamble",  "preamble_bits":  8, "sync_errors": 5, "ber_tolerance_percent": 55, "max_frames": 6},
        # Noisy channel profiles for faded / low-SNR captures
        {"name": "noisy_channel",     "preamble_bits":  0, "sync_errors": 8, "ber_tolerance_percent": 55, "max_frames": 4},
        {"name": "noisy_preamble",    "preamble_bits": 16, "sync_errors": 8, "ber_tolerance_percent": 60, "max_frames": 4},
        # Ultra-relaxed: last resort for captures with badly distorted sync
        {"name": "ultra_noisy",       "preamble_bits":  0, "sync_errors":11, "ber_tolerance_percent": 60, "max_frames": 3},
    )
    BER_TOLERANCE_PERCENT = 50
    CRC_BITS = 16

    def can_handle(self, request: ProtocolDecodeRequest) -> float:
        modulation = (request.modulation_type or "").upper()
        modulation_score = 0.2 if any(name in modulation for name in ("BPSK", "QPSK", "PSK8", "PSK", "OQPSK")) else 0.0

        band_score = 0.0
        if request.center_freq is not None and self.BAND_START_HZ <= float(request.center_freq) <= self.BAND_END_HZ:
            band_score = 0.4

        sync_score = 0.0
        for candidate_bits in self._candidate_bitstreams(request):
            if candidate_bits.size > 0:
                sync_score = max(sync_score, self._best_sync_score(candidate_bits))

        return float(min(1.0, band_score + modulation_score + 0.5 * sync_score))

    def decode(self, request: ProtocolDecodeRequest) -> Dict[str, Any]:
        candidates = self._candidate_bitstreams(request)

        parsed_frames: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        frame_candidates = 0
        frame_locks = 0
        frame_ber_values: List[float] = []

        best_candidate_bits = np.array([], dtype=np.uint8)
        best_candidate_score = 0.0

        for candidate_bits in candidates:
            if candidate_bits.size < len(self.SYNC_WORD):
                continue
            candidate_score = self._best_sync_score(candidate_bits)
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_candidate_bits = candidate_bits

        if best_candidate_bits.size == 0:
            return {
                "matched_protocol": self.plugin_id,
                "protocol_name": self.protocol_name,
                "confidence": 0.0,
                "results": [],
                "artifacts": [],
                **self._build_native_counter_summary(results=[], frame_total=0, frame_locks=0, average_ber=None),
            }

        sw_len = len(self.SYNC_WORD)
        # Track (sync_pos, preamble_bits) already processed to avoid duplicates
        seen_positions: set = set()

        # Outer loop: each profile has its own sync_errors tolerance — looser profiles
        # find positions that strict profiles miss (e.g. Inmarsat-D/IsatM2M with ~8 bit errors).
        for profile in self.FRAME_PROFILES:
            profile_max_errors = int(profile["sync_errors"])
            profile_ber_tol = int(profile["ber_tolerance_percent"])
            profile_preamble = int(profile["preamble_bits"])
            profile_max_frames = int(profile.get("max_frames", 8))

            sync_positions = self._find_sync_positions(
                best_candidate_bits,
                max_errors=profile_max_errors,
                max_frames=profile_max_frames,
            )

            for sync_pos in sync_positions:
                pos_key = (sync_pos, profile_preamble)
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                frame_start = sync_pos + sw_len + profile_preamble
                available = best_candidate_bits.size - frame_start
                if available < 16:
                    continue

                # Accept partial frames — mark them so frame_locked can stay False
                is_partial = available < self.FRAME_LENGTH
                frame_end = frame_start + (self.FRAME_LENGTH if not is_partial else available)

                frame_candidates += 1
                raw_frame_bits = best_candidate_bits[frame_start:frame_end]
                decoded_frame_bits = self._descramble(raw_frame_bits)
                crc_info = self._validate_frame_crc(decoded_frame_bits)
                ber_percent = self._estimate_frame_ber_percent(raw_frame_bits, decoded_frame_bits)
                frame_ber_values.append(ber_percent / 100.0)

                # Measure actual sync quality at this position
                window = best_candidate_bits[sync_pos: sync_pos + sw_len]
                sync_errors_actual = int(np.sum(window != self.SYNC_WORD))
                sync_quality = 1.0 - (sync_errors_actual / sw_len)

                # Data quality of decoded payload
                ones_ratio = float(np.mean(decoded_frame_bits)) if decoded_frame_bits.size > 0 else 0.5
                data_valid = 0.15 <= ones_ratio <= 0.85

                # Primary lock: CRC pass
                crc_locked = crc_info.get("crc_ok") is True
                # Secondary lock: high sync quality + valid data + BER within tolerance + full frame
                secondary_locked = (
                    sync_quality >= 0.85
                    and data_valid
                    and ber_percent <= profile_ber_tol
                    and not is_partial
                )
                frame_locked = crc_locked or secondary_locked

                parsed = self._parse_frame(decoded_frame_bits) or {
                    "packet_type": "frame_candidate",
                    "descriptor": None,
                    "text": "",
                    "is_uncertain": True,
                }
                parsed["ber"] = ber_percent
                parsed["frame_locked"] = bool(frame_locked)
                parsed["crc_ok"] = crc_info.get("crc_ok")
                parsed["crc_available"] = crc_info.get("crc_available", False)
                parsed["crc_expected"] = crc_info.get("crc_expected")
                parsed["crc_computed"] = crc_info.get("crc_computed")
                parsed["crc_algorithm"] = crc_info.get("crc_algorithm")
                parsed["sync_position"] = int(sync_pos)
                parsed["sync_quality"] = float(sync_quality)
                parsed["sync_errors_actual"] = sync_errors_actual
                parsed["frame_profile"] = profile["name"]
                parsed["preamble_bits"] = profile_preamble
                parsed["is_partial"] = bool(is_partial)
                parsed["is_uncertain"] = not frame_locked

                if frame_locked:
                    frame_locks += 1

                parsed_frames.append(parsed)
                text = parsed.get("text", "")
                if text:
                    artifacts.append({
                        "type": "protocol_text",
                        "confidence": float(max(0.35, 1.0 - ber_percent / 100.0)),
                        "payload": {
                            "protocol": self.protocol_name,
                            "packet_type": parsed.get("packet_type"),
                            "text": text,
                            "descriptor": parsed.get("descriptor"),
                            "frame_profile": profile["name"],
                        },
                    })

        if not parsed_frames:
            fallback_pos, fallback_score = self._best_sync_match(best_candidate_bits)
            if fallback_pos is not None or best_candidate_bits.size >= 64:
                frame_start = int(fallback_pos if fallback_pos is not None else 0)
                if fallback_pos is not None:
                    frame_start += sw_len
                frame_end = min(
                    best_candidate_bits.size,
                    frame_start + min(self.FRAME_LENGTH, max(64, best_candidate_bits.size - frame_start)),
                )
                if frame_end > frame_start:
                    raw_frame_bits = best_candidate_bits[frame_start:frame_end]
                    decoded_frame_bits = self._descramble(raw_frame_bits)
                    ber_percent = self._estimate_frame_ber_percent(raw_frame_bits, decoded_frame_bits)
                    crc_info = self._validate_frame_crc(decoded_frame_bits)
                    parsed = self._parse_frame(decoded_frame_bits) or {
                        "packet_type": "frame_candidate",
                        "descriptor": None,
                        "text": "",
                        "is_uncertain": True,
                    }
                    parsed["ber"] = ber_percent
                    parsed["frame_locked"] = False
                    parsed["crc_ok"] = crc_info.get("crc_ok")
                    parsed["crc_available"] = crc_info.get("crc_available", False)
                    parsed["crc_expected"] = crc_info.get("crc_expected")
                    parsed["crc_computed"] = crc_info.get("crc_computed")
                    parsed["crc_algorithm"] = crc_info.get("crc_algorithm")
                    parsed["sync_position"] = int(frame_start)
                    parsed["is_uncertain"] = True
                    parsed["frame_candidate_score"] = float(fallback_score if fallback_pos is not None else 0.05)
                    parsed_frames.append(parsed)
                    artifacts.append({
                        "type": "protocol_frame_candidate",
                        "confidence": float(max(0.15, fallback_score if fallback_pos is not None else 0.05)),
                        "payload": {
                            "protocol": self.protocol_name,
                            "sync_position": int(frame_start),
                            "candidate_score": float(fallback_score if fallback_pos is not None else 0.05),
                            "frame_bits": int(decoded_frame_bits.size),
                            "preview_hex": _bits_to_bytes(decoded_frame_bits[: min(128, decoded_frame_bits.size)]).hex(),
                        },
                    })

        confidence = 0.0
        if parsed_frames:
            confidence = min(0.98, 0.35 + 0.16 * len(parsed_frames) + 0.12 * frame_locks + 0.1 * best_candidate_score)

        average_ber = float(np.mean(frame_ber_values)) if frame_ber_values else None
        native_counters = self._build_native_counter_summary(
            results=parsed_frames,
            frame_total=frame_candidates,
            frame_locks=frame_locks,
            average_ber=average_ber,
        )

        return {
            "matched_protocol": self.plugin_id,
            "protocol_name": self.protocol_name,
            "confidence": float(confidence),
            "results": parsed_frames,
            "artifacts": artifacts,
            **native_counters,
        }

    def _candidate_bitstreams(self, request: ProtocolDecodeRequest) -> List[np.ndarray]:
        """Build a small set of candidate bitstreams from bits and auxiliary audio."""
        candidates: List[np.ndarray] = []

        direct_bits = _as_bits(request.bits)
        if direct_bits.size > 0:
            candidates.append(direct_bits)
            candidates.append(1 - direct_bits)

        audio_bits = _audio_to_bits(request.auxiliary_signal)
        if audio_bits.size > 0:
            candidates.append(audio_bits)
            candidates.append(1 - audio_bits)

        return candidates

    def _frame_profiles(self) -> List[Dict[str, Any]]:
        """Return the Inmarsat family hypotheses used to test sync/preamble alignment."""
        return [dict(profile) for profile in self.FRAME_PROFILES]

    def _best_sync_match(self, bits: np.ndarray) -> tuple[Optional[int], float]:
        """Return the strongest sync match position and score."""
        if bits.size < len(self.SYNC_WORD):
            return None, 0.0

        best_pos: Optional[int] = None
        best_score = 0.0
        sw_len = len(self.SYNC_WORD)
        for index in range(0, bits.size - sw_len + 1):
            window = bits[index : index + sw_len]
            errors = int(np.sum(window != self.SYNC_WORD))
            score = 1.0 - (errors / sw_len)
            if score > best_score:
                best_score = score
                best_pos = index

        return best_pos, float(max(0.0, best_score))

    def _descramble(self, bits: np.ndarray) -> np.ndarray:
        """Descramble Inmarsat-C frame with known LFSR polynomial."""
        state = 0x3FFFFF
        out = np.zeros_like(bits, dtype=np.uint8)
        for i, bit in enumerate(bits):
            lfsr_bit = (state >> 22) & 1
            out[i] = bit ^ lfsr_bit
            feedback = ((state >> 22) ^ (state >> 17)) & 1
            state = ((state << 1) | feedback) & 0x7FFFFF
        return out

    def _estimate_frame_ber_percent(self, raw_bits: np.ndarray, decoded_bits: np.ndarray) -> int:
        """Estimate BER from descrambler delta for internal quality tracking."""
        if raw_bits.size == 0 or decoded_bits.size == 0 or raw_bits.size != decoded_bits.size:
            return 100
        changed = int(np.sum(raw_bits != decoded_bits))
        return int(min(100, round((changed * 100) / raw_bits.size)))

    def _validate_frame_crc(self, frame_bits: np.ndarray) -> Dict[str, Any]:
        """Validate frame CRC using CRC-16/CCITT-FALSE over frame bytes excluding trailer."""
        if frame_bits.size < (8 + self.CRC_BITS):
            return {
                "crc_available": False,
                "crc_ok": None,
                "crc_expected": None,
                "crc_computed": None,
                "crc_algorithm": None,
            }

        frame_bytes = _bits_to_bytes(frame_bits)
        if len(frame_bytes) < 3:
            return {
                "crc_available": False,
                "crc_ok": None,
                "crc_expected": None,
                "crc_computed": None,
                "crc_algorithm": None,
            }

        payload = frame_bytes[:-2]
        expected_crc = int.from_bytes(frame_bytes[-2:], byteorder="big", signed=False)
        computed_crc = self._crc16_ccitt_false(payload)

        return {
            "crc_available": True,
            "crc_ok": bool(computed_crc == expected_crc),
            "crc_expected": expected_crc,
            "crc_computed": computed_crc,
            "crc_algorithm": "CRC-16/CCITT-FALSE",
        }

    def _crc16_ccitt_false(self, data: bytes) -> int:
        """Compute CRC-16/CCITT-FALSE."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def _best_sync_score(self, bits: np.ndarray) -> float:
        if bits.size < len(self.SYNC_WORD):
            return 0.0

        sw_len = len(self.SYNC_WORD)
        best = 0.0
        for i in range(0, bits.size - sw_len + 1):
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_WORD))
            score = 1.0 - (errors / sw_len)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        return float(max(0.0, best))

    def _find_sync_positions(self, bits: np.ndarray, max_errors: int, max_frames: int) -> List[int]:
        positions: List[int] = []
        sw_len = len(self.SYNC_WORD)

        i = 0
        while i <= bits.size - sw_len and len(positions) < max_frames:
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_WORD))
            if errors <= max_errors:
                positions.append(i)
                i += sw_len
                continue
            i += 1

        return positions

    def _parse_frame(self, frame_bits: np.ndarray) -> Optional[Dict[str, Any]]:
        if frame_bits.size < 16:
            return None

        descriptor = int(np.packbits(frame_bits[:8])[0])
        packet_type = self.PACKET_TYPES.get(descriptor, f"Unknown(0x{descriptor:02X})")

        payload_end = -self.CRC_BITS if frame_bits.size >= (8 + self.CRC_BITS) else None
        payload_bits = frame_bits[8:payload_end]
        text = _bits_to_text(payload_bits).strip("\x00 \n\r\t")

        zeros = int(np.sum(frame_bits == 0))
        ones = int(np.sum(frame_bits == 1))
        imbalance = abs(zeros - ones) / max(1, frame_bits.size)

        return {
            "packet_type": packet_type,
            "descriptor": hex(descriptor),
            "text": text,
            "is_uncertain": imbalance > 0.35,
        }


class IridiumBurstProtocolPlugin(ProtocolPlugin):
    """Baseline Iridium burst analysis plugin with native counters contract."""

    plugin_id = "iridium_burst"
    protocol_name = "Iridium Burst Baseline"

    BAND_START_HZ = 1616.0e6
    BAND_END_HZ = 1627.0e6
    GAP_THRESHOLD_BITS = 24
    MIN_BURST_BITS = 96

    def can_handle(self, request: ProtocolDecodeRequest) -> float:
        bits = _as_bits(request.bits)
        if bits.size < self.MIN_BURST_BITS:
            return 0.0

        center_freq = float(request.center_freq) if request.center_freq is not None else 0.0
        in_band = self.BAND_START_HZ <= center_freq <= self.BAND_END_HZ
        band_score = 0.55 if in_band else 0.0

        modulation = (request.modulation_type or "").upper()
        modulation_score = 0.15 if any(name in modulation for name in ("QPSK", "BPSK", "OQPSK")) else 0.0

        bursts = self._split_bursts(bits)
        burst_score = min(0.25, 0.1 * len(bursts)) if bursts else 0.0

        return float(min(0.95, band_score + modulation_score + burst_score))

    def decode(self, request: ProtocolDecodeRequest) -> Dict[str, Any]:
        bits = _as_bits(request.bits)
        bursts = self._split_bursts(bits)

        results: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        ber_values: List[float] = []
        frame_locks = 0

        for index, burst in enumerate(bursts):
            burst_bits = burst["bits"]
            length = int(len(burst_bits))
            if length < self.MIN_BURST_BITS:
                continue

            transitions = int(np.sum(np.diff(burst_bits) != 0)) if length > 1 else 0
            transition_density = float(transitions / max(1, length - 1))
            ones_ratio = float(np.mean(burst_bits)) if length > 0 else 0.0
            validation_error = min(1.0, abs(ones_ratio - 0.5) * 2.0 + abs(transition_density - 0.5))
            ber_proxy = float(max(0.0, min(1.0, validation_error)))
            frame_locked = bool(0.15 <= ones_ratio <= 0.85 and transition_density >= 0.2)

            if frame_locked:
                frame_locks += 1
            ber_values.append(ber_proxy)

            burst_type = "short_burst" if length < 256 else "long_burst"
            preview_hex = _bits_to_bytes(burst_bits[: min(length, 128)]).hex()

            result = {
                "burst_index": int(index),
                "burst_type": burst_type,
                "bit_length": length,
                "sync_position": int(burst["start"]),
                "transition_density": transition_density,
                "ones_ratio": ones_ratio,
                "ber": ber_proxy,
                "frame_locked": frame_locked,
                "crc_ok": None,
                "crc_available": False,
                "preview_hex": preview_hex,
                "is_uncertain": not frame_locked,
            }
            results.append(result)

            artifacts.append(
                {
                    "type": "protocol_burst_log",
                    "confidence": float(max(0.25, 1.0 - ber_proxy)),
                    "payload": {
                        "protocol": self.protocol_name,
                        "burst_type": burst_type,
                        "bit_length": length,
                        "sync_position": int(burst["start"]),
                        "preview_hex": preview_hex,
                    },
                }
            )

        confidence = 0.0
        if results:
            confidence = float(min(0.92, 0.4 + 0.12 * len(results)))

        counters = self._build_native_counter_summary(
            results=results,
            frame_total=len(results),
            frame_locks=frame_locks,
            average_ber=float(np.mean(ber_values)) if ber_values else None,
        )

        return {
            "matched_protocol": self.plugin_id,
            "protocol_name": self.protocol_name,
            "confidence": confidence,
            "results": results,
            "artifacts": artifacts,
            **counters,
        }

    def _split_bursts(self, bits: np.ndarray) -> List[Dict[str, Any]]:
        """Split bitstream into burst candidates using long zero runs as separators."""
        if bits.size == 0:
            return []

        bursts: List[Dict[str, Any]] = []
        current_start = 0
        zero_run = 0

        for index, bit in enumerate(bits):
            if bit == 0:
                zero_run += 1
            else:
                if zero_run >= self.GAP_THRESHOLD_BITS:
                    end = index - zero_run
                    if end - current_start >= self.MIN_BURST_BITS:
                        bursts.append({"start": current_start, "bits": bits[current_start:end]})
                    current_start = index
                zero_run = 0

        final_end = bits.size - zero_run if zero_run >= self.GAP_THRESHOLD_BITS else bits.size
        if final_end - current_start >= self.MIN_BURST_BITS:
            bursts.append({"start": current_start, "bits": bits[current_start:final_end]})

        if not bursts and bits.size >= self.MIN_BURST_BITS:
            bursts.append({"start": 0, "bits": bits})

        return bursts


class MeteorLrptProtocolPlugin(ProtocolPlugin):
    """Meteor LRPT plugin with staged real decode chain and native counter contract."""

    plugin_id = "meteor_lrpt"
    protocol_name = "Meteor LRPT Baseline"

    BAND_START_HZ = 137.0e6
    BAND_END_HZ = 138.0e6
    FRAME_BITS = 1024
    PREVIEW_BYTES = 128
    SYNC_WORD = np.array(
        [
            1, 1, 0, 0, 1, 0, 1, 0,
            0, 1, 1, 1, 0, 0, 1, 1,
            1, 0, 0, 1, 1, 0, 1, 0,
            1, 1, 0, 0, 0, 1, 1, 0,
        ],
        dtype=np.uint8,
    )

    def __init__(self):
        self.decode_chain = MeteorLrptDecodeChain(interleave_depth=4, rs_nsym=16)

    def can_handle(self, request: ProtocolDecodeRequest) -> float:
        bits = _as_bits(request.bits)
        if bits.size < len(self.SYNC_WORD) + self.FRAME_BITS:
            return 0.0

        center_freq = float(request.center_freq) if request.center_freq is not None else 0.0
        in_band = self.BAND_START_HZ <= center_freq <= self.BAND_END_HZ
        band_score = 0.55 if in_band else 0.0

        modulation = (request.modulation_type or "").upper()
        modulation_score = 0.2 if any(name in modulation for name in ("QPSK", "OQPSK")) else 0.0

        sync_score = self._best_sync_score(bits)
        return float(min(0.97, band_score + modulation_score + 0.25 * sync_score))

    def decode(self, request: ProtocolDecodeRequest) -> Dict[str, Any]:
        bits = _as_bits(request.bits)
        sync_positions = self._find_sync_positions(bits, max_errors=3, max_frames=6)

        results: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        frame_locks = 0
        ber_values: List[float] = []
        viterbi_metrics: List[float] = []
        viterbi_soft_metrics: List[float] = []
        viterbi_soft_confidences: List[float] = []
        viterbi_soft_usage: List[float] = []
        rs_corrected_symbols: List[float] = []
        rs_successes = 0
        frame_total = 0

        for frame_index, sync_pos in enumerate(sync_positions):
            frame_start = sync_pos + len(self.SYNC_WORD)
            frame_end = frame_start + self.FRAME_BITS
            if frame_end > bits.size:
                continue

            frame_total += 1
            frame_bits = bits[frame_start:frame_end].astype(np.uint8)

            chain_out = self.decode_chain.decode_frame(frame_bits)
            decoded_bits = _as_bits(chain_out.get("decoded_bits"))
            stages = chain_out.get("stages", {})
            deinterleave_info = stages.get("deinterleave", {})
            viterbi_info = stages.get("viterbi", {})
            rs_info = stages.get("reed_solomon", {})

            ones_ratio = float(np.mean(decoded_bits)) if decoded_bits.size > 0 else 0.0
            transitions = int(np.sum(np.diff(decoded_bits) != 0)) if decoded_bits.size > 1 else 0
            transition_density = float(transitions / max(1, decoded_bits.size - 1))
            ber_proxy = float(min(1.0, abs(ones_ratio - 0.5) * 2.0 + abs(transition_density - 0.5)))
            frame_locked = bool(0.2 <= ones_ratio <= 0.8 and transition_density >= 0.2)

            if frame_locked:
                frame_locks += 1
            ber_values.append(ber_proxy)
            viterbi_metrics.append(float(viterbi_info.get("metric", 1.0)))
            if viterbi_info.get("soft_metric") is not None:
                viterbi_soft_metrics.append(float(viterbi_info.get("soft_metric")))
            if viterbi_info.get("soft_confidence") is not None:
                viterbi_soft_confidences.append(float(viterbi_info.get("soft_confidence")))
            viterbi_soft_usage.append(1.0 if viterbi_info.get("soft_path_used") else 0.0)
            rs_corrected_symbols.append(float(rs_info.get("corrected_symbols", 0.0)))
            if rs_info.get("decode_success"):
                rs_successes += 1

            payload_preview = _bits_to_bytes(decoded_bits[: self.PREVIEW_BYTES * 8]).hex()
            result = {
                "frame_index": int(frame_index),
                "packet_type": "lrpt_frame",
                "sync_position": int(sync_pos),
                "bit_length": int(decoded_bits.size),
                "ber": ber_proxy,
                "frame_locked": frame_locked,
                "crc_ok": bool(rs_info.get("decode_success")),
                "crc_available": bool(rs_info.get("applied", False)),
                "is_uncertain": not frame_locked,
                "preview_hex": payload_preview,
                "decode_chain": {
                    "deinterleave": deinterleave_info,
                    "viterbi": viterbi_info,
                    "reed_solomon": rs_info,
                },
            }
            results.append(result)

            artifacts.append(
                {
                    "type": "protocol_packet_log",
                    "confidence": float(max(0.25, 1.0 - ber_proxy)),
                    "payload": {
                        "protocol": self.protocol_name,
                        "packet_type": "lrpt_frame",
                        "frame_index": int(frame_index),
                        "sync_position": int(sync_pos),
                        "preview_hex": payload_preview,
                    },
                }
            )

        confidence = 0.0
        if results:
            confidence = float(min(0.94, 0.42 + 0.1 * len(results)))

        counters = self._build_native_counter_summary(
            results=results,
            frame_total=frame_total,
            frame_locks=frame_locks,
            average_ber=float(np.mean(ber_values)) if ber_values else None,
        )

        detailed_quality = {
            "deinterleave_depth": int(self.decode_chain.interleave_depth),
            "viterbi_metric_avg": float(np.mean(viterbi_metrics)) if viterbi_metrics else None,
            "viterbi_soft_metric_avg": float(np.mean(viterbi_soft_metrics)) if viterbi_soft_metrics else None,
            "viterbi_soft_confidence_avg": float(np.mean(viterbi_soft_confidences)) if viterbi_soft_confidences else None,
            "viterbi_soft_path_rate": float(np.mean(viterbi_soft_usage)) if viterbi_soft_usage else None,
            "rs_corrected_symbols_avg": float(np.mean(rs_corrected_symbols)) if rs_corrected_symbols else None,
            "rs_decode_success_rate": float(rs_successes / frame_total) if frame_total > 0 else None,
            "decode_chain_depth": "frame_sync->deinterleave->viterbi->reed_solomon",
        }

        return {
            "matched_protocol": self.plugin_id,
            "protocol_name": self.protocol_name,
            "confidence": confidence,
            "results": results,
            "artifacts": artifacts,
            **detailed_quality,
            **counters,
        }

    def _best_sync_score(self, bits: np.ndarray) -> float:
        sw_len = len(self.SYNC_WORD)
        if bits.size < sw_len:
            return 0.0

        best = 0.0
        for i in range(0, bits.size - sw_len + 1):
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_WORD))
            score = 1.0 - (errors / sw_len)
            if score > best:
                best = score
                if best >= 1.0:
                    break

        return float(max(0.0, best))

    def _find_sync_positions(self, bits: np.ndarray, max_errors: int, max_frames: int) -> List[int]:
        positions: List[int] = []
        sw_len = len(self.SYNC_WORD)

        i = 0
        while i <= bits.size - sw_len and len(positions) < max_frames:
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_WORD))
            if errors <= max_errors:
                positions.append(i)
                i += sw_len
                continue
            i += 1

        return positions


class NoaaAptProtocolPlugin(ProtocolPlugin):
    """NOAA APT analog/image baseline using FM-demod surrogate and line sync extraction."""

    plugin_id = "noaa_apt"
    protocol_name = "NOAA APT Baseline"

    BAND_START_HZ = 137.0e6
    BAND_END_HZ = 138.0e6
    LINE_BITS = 832
    MAX_SYNC_ERRORS = 2
    SYNC_PATTERN = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.uint8)

    def can_handle(self, request: ProtocolDecodeRequest) -> float:
        bits = _as_bits(request.bits)
        if bits.size < self.LINE_BITS * 2:
            return 0.0

        center_freq = float(request.center_freq) if request.center_freq is not None else 0.0
        in_band = self.BAND_START_HZ <= center_freq <= self.BAND_END_HZ
        band_score = 0.5 if in_band else 0.0

        modulation = (request.modulation_type or "").upper()
        modulation_score = 0.3 if any(name in modulation for name in ("FM", "WFM", "NFM")) else 0.0

        sync_score = self._estimate_sync_score(bits)
        return float(min(0.96, band_score + modulation_score + 0.2 * sync_score))

    def decode(self, request: ProtocolDecodeRequest) -> Dict[str, Any]:
        bits = _as_bits(request.bits)
        fm_audio = self._fm_demod_surrogate(bits)
        line_sync_hits = self._detect_line_sync(bits, max_lines=24, max_errors=self.MAX_SYNC_ERRORS)
        line_starts = [entry["position"] for entry in line_sync_hits]

        if not line_starts:
            return {
                "matched_protocol": self.plugin_id,
                "protocol_name": self.protocol_name,
                "confidence": 0.0,
                "results": [],
                "artifacts": [],
                **self._build_native_counter_summary(results=[], frame_total=0, frame_locks=0, average_ber=None),
            }

        image_rows: List[np.ndarray] = []
        results: List[Dict[str, Any]] = []
        frame_locks = 0
        ber_values: List[float] = []

        sync_error_values: List[int] = []

        for idx, sync_hit in enumerate(line_sync_hits):
            start = int(sync_hit.get("position", 0))
            sync_errors = int(sync_hit.get("errors", len(self.SYNC_PATTERN)))
            end = start + self.LINE_BITS
            if end > bits.size:
                continue

            line_bits = bits[start:end]
            sync_ok = bool(sync_errors <= 1)
            if sync_ok:
                frame_locks += 1
            sync_error_values.append(sync_errors)

            payload_bits = line_bits[len(self.SYNC_PATTERN) :]
            row = self._line_bits_to_pixels(payload_bits)
            if row.size == 0:
                continue

            image_rows.append(row)

            brightness_var = float(np.std(row.astype(np.float32) / 255.0))
            ber_proxy = float(max(0.0, min(1.0, 1.0 - min(1.0, brightness_var * 2.0))))
            ber_values.append(ber_proxy)

            results.append(
                {
                    "line_index": int(idx),
                    "packet_type": "apt_line",
                    "sync_position": int(start),
                    "bit_length": int(line_bits.size),
                    "ber": ber_proxy,
                    "frame_locked": sync_ok,
                    "crc_ok": None,
                    "crc_available": False,
                    "line_variance": brightness_var,
                    "sync_errors": int(sync_errors),
                    "is_uncertain": not sync_ok,
                }
            )

        if not image_rows:
            return {
                "matched_protocol": self.plugin_id,
                "protocol_name": self.protocol_name,
                "confidence": 0.0,
                "results": results,
                "artifacts": [],
                **self._build_native_counter_summary(
                    results=results,
                    frame_total=len(results),
                    frame_locks=frame_locks,
                    average_ber=float(np.mean(ber_values)) if ber_values else None,
                ),
            }

        image = np.vstack(image_rows)
        image_summary = {
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "mean": float(np.mean(image)),
        }

        artifacts = [
            {
                "type": "image",
                "confidence": float(min(0.93, 0.45 + 0.02 * image.shape[0])),
                "payload": {
                    "protocol": self.protocol_name,
                    "format": "grayscale_8bit",
                    "summary": image_summary,
                    "image_matrix": image.tolist(),
                    "line_sync_count": int(frame_locks),
                    "line_sync_positions": [int(v) for v in line_starts[:16]],
                    "line_sync_errors": [int(entry.get("errors", 0)) for entry in line_sync_hits[:16]],
                    "preview_rows": image[: min(8, image.shape[0]), : min(64, image.shape[1])].tolist(),
                },
            },
            {
                "type": "audio",
                "confidence": 0.55,
                "payload": {
                    "protocol": self.protocol_name,
                    "sample_rate": float(request.sample_rate),
                    "samples": int(fm_audio.size),
                    "rms": float(np.sqrt(np.mean(np.square(fm_audio)))) if fm_audio.size else 0.0,
                },
            },
        ]

        counters = self._build_native_counter_summary(
            results=results,
            frame_total=len(results),
            frame_locks=frame_locks,
            average_ber=float(np.mean(ber_values)) if ber_values else None,
        )

        return {
            "matched_protocol": self.plugin_id,
            "protocol_name": self.protocol_name,
            "confidence": float(min(0.95, 0.5 + 0.03 * len(results))),
            "results": results,
            "artifacts": artifacts,
            "fm_demod_stage": {
                "method": "bit_phase_diff_surrogate",
                "samples": int(fm_audio.size),
            },
            "line_sync_stage": {
                "detected_lines": int(len(results)),
                "locked_lines": int(frame_locks),
                "max_sync_errors": int(self.MAX_SYNC_ERRORS),
                "avg_sync_errors": float(np.mean(sync_error_values)) if sync_error_values else None,
            },
            **counters,
        }

    def _estimate_sync_score(self, bits: np.ndarray) -> float:
        sw_len = len(self.SYNC_PATTERN)
        if bits.size < sw_len:
            return 0.0

        best = 0.0
        step = max(1, sw_len // 2)
        for i in range(0, bits.size - sw_len + 1, step):
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_PATTERN))
            score = 1.0 - (errors / sw_len)
            if score > best:
                best = score

        return float(max(0.0, best))

    def _fm_demod_surrogate(self, bits: np.ndarray) -> np.ndarray:
        if bits.size == 0:
            return np.array([], dtype=np.float32)

        phase_signal = bits.astype(np.float32) * 2.0 - 1.0
        fm = np.diff(phase_signal, prepend=phase_signal[0])
        if fm.size > 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            fm = np.convolve(fm, kernel, mode="same")
        return fm.astype(np.float32)

    def _detect_line_sync(self, bits: np.ndarray, max_lines: int, max_errors: int = 0) -> List[Dict[str, int]]:
        positions: List[Dict[str, int]] = []
        sw_len = len(self.SYNC_PATTERN)
        i = 0

        while i <= bits.size - self.LINE_BITS and len(positions) < max_lines:
            window = bits[i : i + sw_len]
            errors = int(np.sum(window != self.SYNC_PATTERN))
            if errors <= max_errors:
                positions.append({"position": int(i), "errors": int(errors)})
                i += self.LINE_BITS
                continue
            i += 1

        return positions

    def _line_bits_to_pixels(self, payload_bits: np.ndarray) -> np.ndarray:
        if payload_bits.size < 8:
            return np.array([], dtype=np.uint8)

        usable = payload_bits[: (payload_bits.size // 8) * 8]
        bytes_line = np.packbits(usable).astype(np.uint8)
        return bytes_line


def create_default_protocol_registry() -> ProtocolPluginRegistry:
    registry = ProtocolPluginRegistry()
    registry.register(InmarsatProtocolPlugin())
    registry.register(NoaaAptProtocolPlugin())
    registry.register(MeteorLrptProtocolPlugin())
    registry.register(IridiumBurstProtocolPlugin())
    return registry
