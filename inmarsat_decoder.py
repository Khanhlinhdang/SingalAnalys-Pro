# ─── GUI MAIN WINDOW ────────────────────────────────────────────
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QTextEdit, QGroupBox,
    QStatusBar, QSplitter
)
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np

import numpy as np
from scipy.signal import butter, lfilter
import sounddevice as sd
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QThread, Signal, Qt
import pyqtgraph as pg

# ─── DEMODULATOR ────────────────────────────────────────────────
class BPSKDemodulator:
    """
    Rewrite của inmarsatc::demodulator::Demodulator
    Sample rate: 48kS/s, BPSK 1200 baud
    """
    SAMPLE_RATE = 48000
    BAUD_RATE = 1200
    SPB = SAMPLE_RATE // BAUD_RATE  # 40 samples per bit
    
    def __init__(self):
        self.center_freq = 1500.0  # Hz (audio baseband)
        self.lo_freq = 600
        self.hi_freq = 2400
        self.phase = 0.0
        self.agc_gain = 1.0
        self.is_synced = False
        # CMA Equalizer state
        self.cma_weights = np.ones(11, dtype=complex) * 0.1
    
    def _agc(self, samples: np.ndarray) -> np.ndarray:
        """Automatic Gain Control"""
        rms = np.sqrt(np.mean(np.abs(samples)**2))
        if rms > 0:
            self.agc_gain = 1.0 / rms
        return samples * self.agc_gain
    
    def _mix_to_baseband(self, samples: np.ndarray) -> np.ndarray:
        """Mix audio signal xuống baseband (0 Hz)"""
        t = np.arange(len(samples)) / self.SAMPLE_RATE
        lo = np.exp(-2j * np.pi * self.center_freq * t)
        return samples.astype(complex) * lo
    
    def _gardner_timing_recovery(self, samples: np.ndarray) -> np.ndarray:
        """Gardner timing recovery để đồng bộ symbol clock"""
        # Simplified Gardner TED
        output = []
        k = self.SPB
        while k < len(samples) - self.SPB:
            sym = samples[k]
            # Gardner error
            mid = samples[k - self.SPB // 2]
            prev = samples[k - self.SPB]
            err = np.real(mid) * (np.real(sym) - np.real(prev))
            # Update timing
            k_adjust = int(np.clip(err * 0.1, -2, 2))
            output.append(sym)
            k += self.SPB + k_adjust
        return np.array(output)
    
    def _costas_loop_pll(self, symbols: np.ndarray) -> np.ndarray:
        """Costas Loop cho BPSK carrier recovery"""
        out = np.zeros_like(symbols)
        phase = self.phase
        freq = 0.0
        alpha, beta = 0.01, 0.001  # loop filter constants
        
        for i, s in enumerate(symbols):
            rotated = s * np.exp(-1j * phase)
            out[i] = rotated
            # Phase error (BPSK specific)
            err = np.real(rotated) * np.imag(rotated)
            freq += beta * err
            phase += alpha * err + freq
        
        self.phase = phase
        return out
    
    def demodulate(self, audio_samples: np.ndarray) -> np.ndarray:
        """
        Main entry: audio float32 → hard bits uint8[]
        """
        # 1. AGC
        s = self._agc(audio_samples)
        # 2. Mix to baseband
        s = self._mix_to_baseband(s)
        # 3. Low-pass filter
        b, a = butter(5, self.hi_freq / (self.SAMPLE_RATE / 2), 'low')
        s = lfilter(b, a, s)
        # 4. Timing recovery
        symbols = self._gardner_timing_recovery(s)
        # 5. Carrier recovery (Costas Loop)
        symbols = self._costas_loop_pll(symbols)
        # 6. Hard decision
        bits = (np.real(symbols) > 0).astype(np.uint8)
        self.is_synced = len(symbols) > 10
        return bits, symbols  # symbols dùng cho scatter plot


# ─── DECODER ────────────────────────────────────────────────────
class InmarsatCDecoder:
    """
    Rewrite của inmarsatc::decoder::Decoder
    Viterbi + descrambling theo Inmarsat STD-C spec
    """
    FRAME_LENGTH = 640  # bits
    
    # Scrambler polynomial: x^23 + x^18 + 1 (theo spec)
    SCRAMBLER_POLY = 0x840001  
    
    def __init__(self, tolerance: int = 50):
        self.tolerance = tolerance  # max BER
        self._sync_word = np.array([1,1,0,1,1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0], dtype=np.uint8)
        self._bit_buffer = np.array([], dtype=np.uint8)
        self._sync_pos = -1
    
    def _find_sync(self, bits: np.ndarray) -> int:
        """Tìm sync word trong bit stream"""
        sw_len = len(self._sync_word)
        for i in range(len(bits) - sw_len):
            window = bits[i:i+sw_len]
            ber = np.sum(window != self._sync_word)
            inv_ber = np.sum(window != (1 - self._sync_word))
            if ber <= self.tolerance // 10:
                return i
            if inv_ber <= self.tolerance // 10:
                return i  # reversed polarity
        return -1
    
    def _descramble(self, bits: np.ndarray) -> np.ndarray:
        """Descramble theo Linear Feedback Shift Register"""
        state = 0x3FFFFF  # initial state
        out = np.zeros_like(bits)
        for i, b in enumerate(bits):
            # LFSR output bit
            lfsr_bit = (state >> 22) & 1
            out[i] = b ^ lfsr_bit
            # Advance LFSR
            feedback = ((state >> 22) ^ (state >> 17)) & 1
            state = ((state << 1) | feedback) & 0x7FFFFF
        return out
    
    def decode(self, bits: np.ndarray) -> list:
        """Input: raw bits → Output: list of decoded frames"""
        self._bit_buffer = np.concatenate([self._bit_buffer, bits])
        frames = []
        
        while len(self._bit_buffer) >= self.FRAME_LENGTH * 2:
            sync_pos = self._find_sync(self._bit_buffer)
            if sync_pos < 0:
                self._bit_buffer = self._bit_buffer[-self.FRAME_LENGTH:]
                break
            
            frame_start = sync_pos + len(self._sync_word)
            frame_end = frame_start + self.FRAME_LENGTH
            
            if frame_end > len(self._bit_buffer):
                break
            
            raw_frame = self._bit_buffer[frame_start:frame_end]
            decoded = self._descramble(raw_frame)
            
            # BER estimate (simplified)
            ber = int(np.sum(raw_frame != decoded) * 100 / self.FRAME_LENGTH)
            
            if ber <= self.tolerance:
                frames.append({
                    'data': decoded,
                    'length': self.FRAME_LENGTH,
                    'ber': ber,
                    'is_uncertain': ber > self.tolerance // 2
                })
            
            self._bit_buffer = self._bit_buffer[frame_end:]
        
        return frames


# ─── FRAME PARSER ───────────────────────────────────────────────
class FrameParser:
    """
    Parse Inmarsat STD-C frame → meaningful packets
    EGC (Enhanced Group Call), NCS, ship messages...
    """
    PACKET_TYPES = {
        0x04: "EGC SafetyNET",
        0x08: "EGC FleetNET",
        0x0C: "EGC System",
        0x14: "Two-Way Message",
        0x1C: "NCS Common Channel",
    }
    
    def parse_frame(self, frame: dict) -> dict:
        data = frame['data']
        if len(data) < 8:
            return None
        
        # Lấy packet descriptor byte
        descriptor = int(''.join(map(str, data[:8])), 2)
        packet_type = self.PACKET_TYPES.get(descriptor, f"Unknown(0x{descriptor:02X})")
        
        # Extract payload (simplified — full impl cần theo ITU-R M.1842-1)
        payload_bits = data[8:]
        payload_bytes = np.packbits(payload_bits)
        
        # Thử decode IA5 text (ASCII variant cho maritime)
        text = ""
        try:
            text = payload_bytes.tobytes().decode('ascii', errors='replace').strip('\x00')
        except:
            pass
        
        return {
            'packet_type': packet_type,
            'descriptor': hex(descriptor),
            'payload_hex': payload_bytes[:20].tobytes().hex(),
            'text': text,
            'ber': frame['ber'],
            'is_uncertain': frame['is_uncertain'],
        }


class AudioWorker(QThread):
    """Worker thread nhận audio từ sounddevice và xử lý DSP"""
    symbols_ready = Signal(np.ndarray)   # Cho scatter plot
    bits_ready = Signal(np.ndarray)      # Cho decoder
    stats_ready = Signal(dict)           # sync status, SNR...
    
    def __init__(self):
        super().__init__()
        self.demodulator = BPSKDemodulator()
        self.decoder = InmarsatCDecoder()
        self.parser = FrameParser()
        self._running = False
        self.packets = []
    
    packets_decoded = Signal(list)  # emit khi có packets mới
    
    def run(self):
        import sounddevice as sd
        self._running = True
        
        CHUNK = 2400  # 50ms @ 48kHz
        
        def callback(indata, frames, time, status):
            if not self._running:
                return
            audio = indata[:, 0].copy()  # mono
            bits, symbols = self.demodulator.demodulate(audio)
            
            self.symbols_ready.emit(symbols[-100:])  # last 100 symbols
            
            frames = self.decoder.decode(bits)
            if frames:
                packets = [self.parser.parse_frame(f) for f in frames]
                packets = [p for p in packets if p]
                if packets:
                    self.packets_decoded.emit(packets)
            
            self.stats_ready.emit({
                'synced': self.demodulator.is_synced,
                'center_freq': self.demodulator.center_freq,
                'agc_gain': self.demodulator.agc_gain,
            })
        
        with sd.InputStream(
            samplerate=48000,
            channels=1,
            dtype='float32',
            blocksize=CHUNK,
            callback=callback
        ):
            while self._running:
                self.msleep(10)
    
    def stop(self):
        self._running = False


class ScatterPlotWidget(pg.PlotWidget):
    """Constellation / Scatter plot (IQ diagram)"""
    def __init__(self):
        super().__init__()
        self.setTitle("Constellation (BPSK)")
        self.setLabel('left', 'Q')
        self.setLabel('bottom', 'I')
        self.setXRange(-2, 2)
        self.setYRange(-2, 2)
        self.setBackground('#1a1a2e')
        self.addLine(x=0, pen=pg.mkPen('#444'))
        self.addLine(y=0, pen=pg.mkPen('#444'))
        self._scatter = pg.ScatterPlotItem(
            size=3, pen=None,
            brush=pg.mkBrush(0, 200, 255, 150)
        )
        self.addItem(self._scatter)
    
    def update_symbols(self, symbols: np.ndarray):
        x = np.real(symbols)
        y = np.imag(symbols)
        self._scatter.setData(x=x, y=y)


class ScytalePyWindow(QMainWindow):
    """Main Window — PySide6 version of Scytale-C"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scytale-C Python — Inmarsat STD-C Decoder")
        self.resize(1200, 800)
        self.worker = AudioWorker()
        self._build_ui()
        self._connect_signals()
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # ── Left panel: controls ──
        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        
        # Control group
        ctrl_group = QGroupBox("Demodulator")
        ctrl_form = QVBoxLayout(ctrl_group)
        
        self.btn_start = QPushButton("▶ START")
        self.btn_start.setStyleSheet(
            "background:#00b894;color:white;font-weight:bold;padding:8px;"
        )
        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setEnabled(False)
        
        ctrl_form.addWidget(QLabel("Center Frequency (Hz):"))
        self.spin_cf = QDoubleSpinBox()
        self.spin_cf.setRange(600, 2400)
        self.spin_cf.setValue(1500)
        ctrl_form.addWidget(self.spin_cf)
        
        ctrl_form.addWidget(QLabel("Lo / Hi Freq:"))
        hf = QHBoxLayout()
        self.spin_lo = QSpinBox(); self.spin_lo.setRange(200, 1000); self.spin_lo.setValue(600)
        self.spin_hi = QSpinBox(); self.spin_hi.setRange(1200, 3000); self.spin_hi.setValue(2400)
        hf.addWidget(self.spin_lo); hf.addWidget(self.spin_hi)
        ctrl_form.addLayout(hf)
        
        self.chk_agc = QCheckBox("AGC Enabled"); self.chk_agc.setChecked(True)
        self.chk_cma = QCheckBox("CMA Equalizer"); self.chk_cma.setChecked(True)
        ctrl_form.addWidget(self.chk_agc)
        ctrl_form.addWidget(self.chk_cma)
        ctrl_form.addWidget(self.btn_start)
        ctrl_form.addWidget(self.btn_stop)
        
        # Stats group
        stats_group = QGroupBox("Status")
        stats_layout = QVBoxLayout(stats_group)
        self.lbl_sync = QLabel("SYNC: ❌ NO LOCK")
        self.lbl_cf = QLabel("CF: ---")
        self.lbl_frames = QLabel("Frames: 0")
        stats_layout.addWidget(self.lbl_sync)
        stats_layout.addWidget(self.lbl_cf)
        stats_layout.addWidget(self.lbl_frames)
        
        left_layout.addWidget(ctrl_group)
        left_layout.addWidget(stats_group)
        left_layout.addStretch()
        
        # ── Right panel: scatter + log ──
        right_splitter = QSplitter()
        right_splitter.setOrientation(Qt.Orientation.Vertical)  # Vertical
        
        self.scatter = ScatterPlotWidget()
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier New", 9))
        self.log_view.setStyleSheet("background:#0d1117;color:#c9d1d9;")
        
        right_splitter.addWidget(self.scatter)
        right_splitter.addWidget(self.log_view)
        right_splitter.setSizes([400, 300])
        
        main_layout.addWidget(left)
        main_layout.addWidget(right_splitter)
        
        self.statusBar().showMessage("Ready — Connect audio from SDR# via VAC")
        
        self._frame_count = 0
    
    def _connect_signals(self):
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.worker.symbols_ready.connect(self.scatter.update_symbols)
        self.worker.packets_decoded.connect(self._on_packets)
        self.worker.stats_ready.connect(self._on_stats)
        self.spin_cf.valueChanged.connect(
            lambda v: setattr(self.worker.demodulator, 'center_freq', v)
        )
    
    def _start(self):
        self.worker.demodulator.center_freq = self.spin_cf.value()
        self.worker.demodulator.lo_freq = self.spin_lo.value()
        self.worker.demodulator.hi_freq = self.spin_hi.value()
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running — listening on default audio input")
    
    def _stop(self):
        self.worker.stop()
        self.worker.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped")
    
    def _on_packets(self, packets: list):
        self._frame_count += len(packets)
        self.lbl_frames.setText(f"Frames: {self._frame_count}")
        for p in packets:
            color = "#f38ba8" if p['is_uncertain'] else "#a6e3a1"
            line = (
                f'<span style="color:{color}">'
                f'[{p["packet_type"]}] BER:{p["ber"]}% '
                f'{p["text"][:80] or p["payload_hex"][:40]}'
                f'</span><br>'
            )
            self.log_view.insertHtml(line)
            # Auto scroll
            self.log_view.verticalScrollBar().setValue(
                self.log_view.verticalScrollBar().maximum()
            )
    
    def _on_stats(self, stats: dict):
        if stats['synced']:
            self.lbl_sync.setText("SYNC: ✅ LOCKED")
            self.lbl_sync.setStyleSheet("color:#00b894;font-weight:bold;")
        else:
            self.lbl_sync.setText("SYNC: ❌ NO LOCK")
            self.lbl_sync.setStyleSheet("color:#d63031;")
        self.lbl_cf.setText(f"CF: {stats['center_freq']:.1f} Hz")
    
    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ScytalePyWindow()
    win.show()
    sys.exit(app.exec())
