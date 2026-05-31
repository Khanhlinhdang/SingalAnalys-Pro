"""Meteor LRPT staged decode chain with real deinterleave, Viterbi, and Reed-Solomon operations."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class MeteorLrptDecodeChain:
    """Decode chain utilities for Meteor LRPT baseline processing."""

    def __init__(self, interleave_depth: int = 4, rs_nsym: int = 16):
        self.interleave_depth = max(2, int(interleave_depth))
        self.rs_nsym = max(2, int(rs_nsym))
        self.constraint_length = 7
        self.generators = (0o171, 0o133)

        self._gf_prim = 0x11D
        self._gf_exp = np.zeros(512, dtype=np.uint16)
        self._gf_log = np.zeros(256, dtype=np.int16)
        self._init_gf_tables()

        self._trellis_next, self._trellis_out = self._build_viterbi_trellis(
            self.constraint_length,
            self.generators,
        )

    def decode_frame(self, bits: np.ndarray, soft_symbols: np.ndarray | None = None) -> Dict[str, Any]:
        norm_bits = self._normalize_bits(bits)

        deinterleaved = self.deinterleave(norm_bits, self.interleave_depth)
        hard_metrics = self._run_viterbi_hard(deinterleaved)
        soft_input = self._prepare_soft_symbols(deinterleaved, soft_symbols)
        soft_metrics = self.viterbi_decode_soft(soft_input)

        viterbi_bits = soft_metrics.get("decoded_bits")
        if viterbi_bits is None or np.asarray(viterbi_bits).size == 0:
            viterbi_bits = hard_metrics.get("decoded_bits", np.array([], dtype=np.uint8))
            active_metric = float(hard_metrics.get("metric", 1.0))
            soft_confidence = 0.0
            soft_metric = None
            soft_path = False
        else:
            viterbi_bits = np.asarray(viterbi_bits, dtype=np.uint8)
            soft_metric = float(soft_metrics.get("path_metric", 0.0))
            soft_confidence = float(soft_metrics.get("soft_confidence", 0.0))
            active_metric = float(self._estimate_viterbi_metric(deinterleaved, viterbi_bits))
            soft_path = True

        rs_input_bytes = self.bits_to_bytes(viterbi_bits)
        rs_output = self.rs_decode(rs_input_bytes, self.rs_nsym)
        rs_message = rs_output.get("message", np.array([], dtype=np.uint8))
        message_bits = np.unpackbits(rs_message).astype(np.uint8) if rs_message.size else np.array([], dtype=np.uint8)

        return {
            "decoded_bits": message_bits,
            "decoded_bytes": rs_message,
            "stages": {
                "deinterleave": {
                    "applied": True,
                    "depth": int(self.interleave_depth),
                    "input_bits": int(norm_bits.size),
                    "output_bits": int(deinterleaved.size),
                },
                "viterbi": {
                    "applied": True,
                    "constraint_length": int(self.constraint_length),
                    "code_rate": "1/2",
                    "input_bits": int(deinterleaved.size),
                    "output_bits": int(viterbi_bits.size),
                    "metric": active_metric,
                    "hard_metric": float(hard_metrics.get("metric", 1.0)),
                    "soft_metric": soft_metric,
                    "soft_confidence": soft_confidence,
                    "soft_path_used": soft_path,
                },
                "reed_solomon": {
                    "applied": True,
                    "nsym": int(self.rs_nsym),
                    "input_bytes": int(rs_input_bytes.size),
                    "output_bytes": int(rs_message.size),
                    **rs_output,
                },
            },
        }

    def encode_for_test(self, payload_bytes: np.ndarray) -> np.ndarray:
        payload = np.asarray(payload_bytes, dtype=np.uint8)
        rs_codeword = self.rs_encode(payload, self.rs_nsym)
        bits = np.unpackbits(rs_codeword).astype(np.uint8)
        conv = self.conv_encode(bits)
        interleaved = self.interleave(conv, self.interleave_depth)
        return interleaved

    def interleave(self, bits: np.ndarray, depth: int) -> np.ndarray:
        arr = self._normalize_bits(bits)
        if arr.size == 0:
            return arr

        rows = max(2, int(depth))
        pad = (-arr.size) % rows
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])

        matrix = arr.reshape(rows, -1)
        return matrix.T.flatten().astype(np.uint8)

    def deinterleave(self, bits: np.ndarray, depth: int) -> np.ndarray:
        arr = self._normalize_bits(bits)
        if arr.size == 0:
            return arr

        rows = max(2, int(depth))
        pad = (-arr.size) % rows
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])

        matrix = arr.reshape(-1, rows)
        return matrix.T.flatten().astype(np.uint8)

    def conv_encode(self, bits: np.ndarray) -> np.ndarray:
        arr = self._normalize_bits(bits)
        state = 0
        k = self.constraint_length
        out = []

        for bit in np.concatenate([arr, np.zeros(k - 1, dtype=np.uint8)]):
            bit_i = int(bit)
            state = ((state << 1) | bit_i) & ((1 << k) - 1)
            g1 = self._parity(state & self.generators[0])
            g2 = self._parity(state & self.generators[1])
            out.extend([g1, g2])

        return np.asarray(out, dtype=np.uint8)

    def viterbi_decode_hard(self, encoded_bits: np.ndarray) -> np.ndarray:
        return self._run_viterbi_hard(encoded_bits).get("decoded_bits", np.array([], dtype=np.uint8))

    def _run_viterbi_hard(self, encoded_bits: np.ndarray) -> Dict[str, Any]:
        bits = self._normalize_bits(encoded_bits)
        if bits.size < 2:
            return {"decoded_bits": np.array([], dtype=np.uint8), "metric": 1.0}

        if bits.size % 2 != 0:
            bits = bits[:-1]

        symbols = bits.reshape(-1, 2)
        n_steps = symbols.shape[0]
        n_states = self._trellis_next.shape[0]

        inf = 1e9
        metrics = np.full((n_steps + 1, n_states), inf, dtype=np.float64)
        prev_state = np.full((n_steps, n_states), -1, dtype=np.int16)
        prev_input = np.full((n_steps, n_states), 0, dtype=np.uint8)
        metrics[0, 0] = 0.0

        for t in range(n_steps):
            rx0, rx1 = int(symbols[t, 0]), int(symbols[t, 1])
            for s in range(n_states):
                current = metrics[t, s]
                if current >= inf:
                    continue

                for u in (0, 1):
                    ns = int(self._trellis_next[s, u])
                    out_pair = int(self._trellis_out[s, u])
                    ex0 = (out_pair >> 1) & 1
                    ex1 = out_pair & 1
                    branch = (rx0 != ex0) + (rx1 != ex1)
                    cand = current + branch
                    if cand < metrics[t + 1, ns]:
                        metrics[t + 1, ns] = cand
                        prev_state[t, ns] = s
                        prev_input[t, ns] = u

        end_state = int(np.argmin(metrics[-1]))
        decoded = np.zeros(n_steps, dtype=np.uint8)
        state = end_state
        for t in range(n_steps - 1, -1, -1):
            decoded[t] = prev_input[t, state]
            ps = prev_state[t, state]
            if ps < 0:
                break
            state = int(ps)

        tail = self.constraint_length - 1
        if decoded.size > tail:
            decoded = decoded[:-tail]

        decoded = decoded.astype(np.uint8)
        return {
            "decoded_bits": decoded,
            "metric": float(self._estimate_viterbi_metric(bits, decoded)),
        }

    def viterbi_decode_soft(self, soft_symbols: np.ndarray) -> Dict[str, Any]:
        soft = np.asarray(soft_symbols, dtype=np.float32).flatten()
        if soft.size < 2:
            return {
                "decoded_bits": np.array([], dtype=np.uint8),
                "path_metric": 0.0,
                "soft_confidence": 0.0,
            }

        if soft.size % 2 != 0:
            soft = soft[:-1]

        symbols = soft.reshape(-1, 2)
        n_steps = symbols.shape[0]
        n_states = self._trellis_next.shape[0]

        inf = 1e12
        metrics = np.full((n_steps + 1, n_states), inf, dtype=np.float64)
        prev_state = np.full((n_steps, n_states), -1, dtype=np.int16)
        prev_input = np.full((n_steps, n_states), 0, dtype=np.uint8)
        metrics[0, 0] = 0.0

        step_margins = []
        for t in range(n_steps):
            rx0, rx1 = float(symbols[t, 0]), float(symbols[t, 1])
            for s in range(n_states):
                current = metrics[t, s]
                if current >= inf:
                    continue

                for u in (0, 1):
                    ns = int(self._trellis_next[s, u])
                    out_pair = int(self._trellis_out[s, u])
                    ex0 = 1.0 if ((out_pair >> 1) & 1) else -1.0
                    ex1 = 1.0 if (out_pair & 1) else -1.0
                    branch = (rx0 - ex0) * (rx0 - ex0) + (rx1 - ex1) * (rx1 - ex1)
                    cand = current + branch
                    if cand < metrics[t + 1, ns]:
                        metrics[t + 1, ns] = cand
                        prev_state[t, ns] = s
                        prev_input[t, ns] = u

            finite_metrics = metrics[t + 1][np.isfinite(metrics[t + 1])]
            if finite_metrics.size >= 2:
                sorted_metrics = np.sort(finite_metrics)
                step_margins.append(float(sorted_metrics[1] - sorted_metrics[0]))

        end_state = int(np.argmin(metrics[-1]))
        decoded = np.zeros(n_steps, dtype=np.uint8)
        state = end_state
        for t in range(n_steps - 1, -1, -1):
            decoded[t] = prev_input[t, state]
            ps = prev_state[t, state]
            if ps < 0:
                break
            state = int(ps)

        tail = self.constraint_length - 1
        if decoded.size > tail:
            decoded = decoded[:-tail]

        margin_avg = float(np.mean(step_margins)) if step_margins else 0.0
        soft_confidence = float(min(1.0, margin_avg / 3.5))
        return {
            "decoded_bits": decoded.astype(np.uint8),
            "path_metric": float(np.min(metrics[-1])),
            "soft_confidence": soft_confidence,
        }

    def rs_encode(self, message: np.ndarray, nsym: int) -> np.ndarray:
        msg = np.asarray(message, dtype=np.uint8)
        ns = int(max(2, nsym))
        gen = self._rs_generator_poly(ns)

        out = np.concatenate([msg, np.zeros(ns, dtype=np.uint8)]).astype(np.uint8)
        for i in range(msg.size):
            coef = int(out[i])
            if coef == 0:
                continue
            for j in range(gen.size):
                out[i + j] ^= self._gf_mul(int(gen[j]), coef)

        parity = out[-ns:]
        return np.concatenate([msg, parity]).astype(np.uint8)

    def rs_decode(self, codeword: np.ndarray, nsym: int) -> Dict[str, Any]:
        cw = np.asarray(codeword, dtype=np.uint8).copy()
        ns = int(max(2, nsym))
        if cw.size <= ns:
            return {
                "decode_success": False,
                "corrected_symbols": 0,
                "syndrome_weight": 0,
                "message": np.array([], dtype=np.uint8),
                "algorithm": "rs_bm_forney",
            }

        synd = self._rs_calc_syndromes(cw, ns)
        syndrome_weight = int(np.count_nonzero(synd))
        if np.all(synd == 0):
            return {
                "decode_success": True,
                "corrected_symbols": 0,
                "syndrome_weight": 0,
                "message": cw[:-ns],
                "algorithm": "rs_bm_forney",
            }

        err_loc = self._rs_find_error_locator(synd, ns)
        err_count = max(0, len(err_loc) - 1)

        if err_count == 0 or (2 * err_count) > ns:
            return {
                "decode_success": False,
                "corrected_symbols": 0,
                "syndrome_weight": syndrome_weight,
                "message": cw[:-ns],
                "algorithm": "rs_bm_forney",
            }

        err_pos = self._rs_find_errors(err_loc, len(cw))
        if err_pos is None or len(err_pos) != err_count:
            return {
                "decode_success": False,
                "corrected_symbols": 0,
                "syndrome_weight": syndrome_weight,
                "message": cw[:-ns],
                "algorithm": "rs_bm_forney",
            }

        fixed = self._rs_correct_errata(cw.copy(), err_pos, err_loc, synd)
        if fixed is None:
            return {
                "decode_success": False,
                "corrected_symbols": 0,
                "syndrome_weight": syndrome_weight,
                "message": cw[:-ns],
                "algorithm": "rs_bm_forney",
            }

        post_synd = self._rs_calc_syndromes(fixed, ns)
        success = bool(np.all(post_synd == 0))
        corrected_symbols = int(len(err_pos)) if success else 0
        return {
            "decode_success": success,
            "corrected_symbols": corrected_symbols,
            "syndrome_weight": syndrome_weight,
            "message": fixed[:-ns] if success else cw[:-ns],
            "algorithm": "rs_bm_forney",
        }

    def bits_to_bytes(self, bits: np.ndarray) -> np.ndarray:
        arr = self._normalize_bits(bits)
        if arr.size < 8:
            return np.array([], dtype=np.uint8)
        usable = arr[: (arr.size // 8) * 8]
        return np.packbits(usable).astype(np.uint8)

    def _normalize_bits(self, bits: np.ndarray) -> np.ndarray:
        arr = np.asarray(bits).astype(np.uint8).flatten()
        return arr & 1

    def _estimate_viterbi_metric(self, encoded_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
        if encoded_bits.size == 0 or decoded_bits.size == 0:
            return 1.0
        expected = self.conv_encode(decoded_bits)
        compare_len = min(expected.size, encoded_bits.size)
        if compare_len == 0:
            return 1.0
        mismatches = int(np.sum(expected[:compare_len] != encoded_bits[:compare_len]))
        return float(mismatches / compare_len)

    def _prepare_soft_symbols(self, encoded_bits: np.ndarray, soft_symbols: np.ndarray | None) -> np.ndarray:
        if soft_symbols is not None:
            soft = np.asarray(soft_symbols, dtype=np.float32).flatten()
            if soft.size >= 2:
                return np.clip(soft, -1.5, 1.5)

        hard = self._normalize_bits(encoded_bits)
        if hard.size == 0:
            return np.array([], dtype=np.float32)
        # Surrogate LLR-like mapping to enable soft-path metrics even with hard decisions.
        return np.where(hard > 0, 0.9, -0.9).astype(np.float32)

    def _build_viterbi_trellis(self, constraint_length: int, generators: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        k = int(constraint_length)
        n_states = 1 << (k - 1)
        next_state = np.zeros((n_states, 2), dtype=np.uint8)
        outputs = np.zeros((n_states, 2), dtype=np.uint8)

        for state in range(n_states):
            for input_bit in (0, 1):
                reg = (state << 1) | input_bit
                g1 = self._parity(reg & int(generators[0]))
                g2 = self._parity(reg & int(generators[1]))
                ns = reg & (n_states - 1)
                next_state[state, input_bit] = ns
                outputs[state, input_bit] = (g1 << 1) | g2

        return next_state, outputs

    def _parity(self, value: int) -> int:
        return int(bin(value).count("1") & 1)

    def _init_gf_tables(self) -> None:
        x = 1
        for i in range(255):
            self._gf_exp[i] = x
            self._gf_log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self._gf_prim
        for i in range(255, 512):
            self._gf_exp[i] = self._gf_exp[i - 255]

    def _gf_mul(self, x: int, y: int) -> int:
        if x == 0 or y == 0:
            return 0
        return int(self._gf_exp[int(self._gf_log[x]) + int(self._gf_log[y])])

    def _gf_div(self, x: int, y: int) -> int:
        if y == 0:
            raise ZeroDivisionError("GF division by zero")
        if x == 0:
            return 0
        return int(self._gf_exp[(int(self._gf_log[x]) + 255 - int(self._gf_log[y])) % 255])

    def _gf_inverse(self, x: int) -> int:
        if x == 0:
            raise ZeroDivisionError("GF inverse of zero")
        return int(self._gf_exp[255 - int(self._gf_log[x])])

    def _gf_pow(self, x: int, power: int) -> int:
        if power == 0:
            return 1
        if x == 0:
            return 0
        return int(self._gf_exp[(int(self._gf_log[x]) * power) % 255])

    def _poly_mul(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        out = np.zeros(p.size + q.size - 1, dtype=np.uint8)
        for i, coef_p in enumerate(p):
            if coef_p == 0:
                continue
            for j, coef_q in enumerate(q):
                if coef_q == 0:
                    continue
                out[i + j] ^= self._gf_mul(int(coef_p), int(coef_q))
        return out

    def _poly_add(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        max_len = max(p.size, q.size)
        out = np.zeros(max_len, dtype=np.uint8)
        out[-p.size :] ^= p
        out[-q.size :] ^= q
        return out

    def _poly_scale(self, p: np.ndarray, x: int) -> np.ndarray:
        if x == 0:
            return np.zeros_like(p)
        return np.array([self._gf_mul(int(coef), int(x)) for coef in p], dtype=np.uint8)

    def _poly_eval(self, poly: np.ndarray, x: int) -> int:
        y = 0
        for coef in poly:
            y = self._gf_mul(y, x) ^ int(coef)
        return int(y)

    def _rs_generator_poly(self, nsym: int) -> np.ndarray:
        gen = np.array([1], dtype=np.uint8)
        for i in range(nsym):
            gen = self._poly_mul(gen, np.array([1, self._gf_pow(2, i)], dtype=np.uint8))
        return gen

    def _rs_calc_syndromes(self, msg: np.ndarray, nsym: int) -> np.ndarray:
        synd = np.zeros(nsym, dtype=np.uint8)
        for i in range(nsym):
            synd[i] = self._poly_eval(msg, self._gf_pow(2, i))
        return synd

    def _rs_find_error_locator(self, synd: np.ndarray, nsym: int) -> np.ndarray:
        err_loc = np.array([1], dtype=np.uint8)
        old_loc = np.array([1], dtype=np.uint8)

        for i in range(nsym):
            old_loc = np.append(old_loc, np.uint8(0))

            delta = int(synd[i])
            for j in range(1, err_loc.size):
                if i - j < 0:
                    continue
                delta ^= self._gf_mul(int(err_loc[-(j + 1)]), int(synd[i - j]))

            if delta == 0:
                continue

            if old_loc.size > err_loc.size:
                new_loc = self._poly_scale(old_loc, delta)
                old_loc = self._poly_scale(err_loc, self._gf_inverse(delta))
                err_loc = new_loc

            err_loc = self._poly_add(err_loc, self._poly_scale(old_loc, delta))

        while err_loc.size > 1 and err_loc[0] == 0:
            err_loc = err_loc[1:]
        return err_loc.astype(np.uint8)

    def _rs_find_errors(self, err_loc: np.ndarray, nmess: int) -> np.ndarray | None:
        errs = err_loc.size - 1
        if errs <= 0:
            return np.array([], dtype=np.int32)

        err_pos = []
        for i in range(nmess):
            # Chien search in inverse-power domain for current coefficient convention.
            if self._poly_eval(err_loc, self._gf_pow(2, -i)) == 0:
                err_pos.append(nmess - 1 - i)

        if len(err_pos) != errs:
            return None
        return np.asarray(err_pos, dtype=np.int32)

    def _rs_find_error_evaluator(self, synd: np.ndarray, err_loc: np.ndarray, nsym: int) -> np.ndarray:
        synd_poly = np.concatenate([synd, np.array([0], dtype=np.uint8)])
        prod = self._poly_mul(synd_poly, err_loc)
        # Keep remainder modulo x^(nsym+1)
        return prod[-(nsym + 1) :].astype(np.uint8)

    def _rs_correct_errata(
        self,
        msg: np.ndarray,
        err_pos: np.ndarray,
        err_loc: np.ndarray,
        syndromes: np.ndarray,
    ) -> np.ndarray | None:
        if err_pos.size == 0:
            return msg

        if err_pos.size > syndromes.size:
            return None

        nmess = msg.size
        errs = int(err_pos.size)
        locator_powers = [nmess - 1 - int(p) for p in err_pos]
        x_values = [self._gf_pow(2, loc) for loc in locator_powers]

        matrix = np.zeros((errs, errs), dtype=np.uint8)
        rhs = np.array(syndromes[:errs], dtype=np.uint8)

        for row in range(errs):
            for col, x_val in enumerate(x_values):
                matrix[row, col] = np.uint8(self._gf_pow(int(x_val), row))

        magnitudes = self._gf_solve_linear_system(matrix, rhs)
        if magnitudes is None:
            return None

        corrected = msg.copy()
        for i, pos in enumerate(err_pos):
            corrected[int(pos)] ^= np.uint8(magnitudes[i])

        return corrected

    def _gf_solve_linear_system(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
        n_rows, n_cols = matrix.shape
        if n_rows != n_cols or rhs.size != n_rows:
            return None

        a = matrix.copy().astype(np.uint8)
        b = rhs.copy().astype(np.uint8)
        n = n_rows

        for col in range(n):
            pivot = None
            for row in range(col, n):
                if int(a[row, col]) != 0:
                    pivot = row
                    break
            if pivot is None:
                return None

            if pivot != col:
                a[[col, pivot]] = a[[pivot, col]]
                b[[col, pivot]] = b[[pivot, col]]

            inv_pivot = self._gf_inverse(int(a[col, col]))
            for j in range(col, n):
                a[col, j] = np.uint8(self._gf_mul(int(a[col, j]), inv_pivot))
            b[col] = np.uint8(self._gf_mul(int(b[col]), inv_pivot))

            for row in range(n):
                if row == col:
                    continue
                factor = int(a[row, col])
                if factor == 0:
                    continue
                for j in range(col, n):
                    a[row, j] = np.uint8(int(a[row, j]) ^ self._gf_mul(factor, int(a[col, j])))
                b[row] = np.uint8(int(b[row]) ^ self._gf_mul(factor, int(b[col])))

        return b

    def _rs_try_single_symbol_correction(self, msg: np.ndarray, nsym: int) -> np.ndarray | None:
        original = msg.copy()
        for pos in range(msg.size):
            old_val = int(msg[pos])
            for candidate in range(256):
                if candidate == old_val:
                    continue
                msg[pos] = np.uint8(candidate)
                synd = self._rs_calc_syndromes(msg, nsym)
                if np.all(synd == 0):
                    return msg.copy()
            msg[pos] = np.uint8(old_val)

        msg[:] = original
        return None
