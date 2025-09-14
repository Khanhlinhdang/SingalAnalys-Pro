
"""
Advanced SDR Application with PySide6, PyQtGraph, and UHD
Tính năng: kết nối USRP N210/X310, hiển thị spectrogram, demodulation, signal detection
"""

import sys
import numpy as np
import time
import struct
from threading import Thread, Event
from queue import Queue

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
                               QComboBox, QSlider, QProgressBar, QTextEdit, QTabWidget,
                               QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph import ImageItem, PlotWidget
import uhd

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, welch


class USRPInterface(QObject):
    """Interface class for USRP N210/X310 connection and control"""

    data_ready = Signal(np.ndarray)  # Signal for new IQ data
    status_changed = Signal(str)     # Signal for status updates

    def __init__(self):
        super().__init__()
        self.usrp = None
        self.is_connected = False
        self.is_receiving = False
        self.center_freq = 100e6  # 100 MHz default
        self.sample_rate = 1e6    # 1 MHz default
        self.gain = 50            # dB
        self.recv_thread = None
        self.stop_event = Event()

    def connect_usrp(self, device_args=""):
        """Connect to USRP device"""
        try:
            if device_args:
                self.usrp = uhd.usrp.MultiUSRP(device_args)
            else:
                # Try to find any available USRP
                self.usrp = uhd.usrp.MultiUSRP()

            # Set initial parameters
            self.usrp.set_rx_rate(self.sample_rate)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.center_freq))
            self.usrp.set_rx_gain(self.gain)

            self.is_connected = True
            self.status_changed.emit("Connected to USRP successfully")
            return True

        except Exception as e:
            self.status_changed.emit(f"Failed to connect: {str(e)}")
            return False

    def disconnect_usrp(self):
        """Disconnect from USRP"""
        self.stop_receiving()
        self.usrp = None
        self.is_connected = False
        self.status_changed.emit("Disconnected from USRP")

    def set_frequency(self, freq_hz):
        """Set center frequency"""
        if self.usrp and self.is_connected:
            self.center_freq = freq_hz
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq_hz))

    def set_sample_rate(self, rate_hz):
        """Set sample rate"""
        if self.usrp and self.is_connected:
            self.sample_rate = rate_hz
            self.usrp.set_rx_rate(rate_hz)

    def set_gain(self, gain_db):
        """Set RX gain"""
        if self.usrp and self.is_connected:
            self.gain = gain_db
            self.usrp.set_rx_gain(gain_db)

    def start_receiving(self, num_samples=1024):
        """Start receiving IQ data in separate thread"""
        if not self.is_connected:
            return False

        self.stop_event.clear()
        self.recv_thread = Thread(target=self._receive_worker, args=(num_samples,))
        self.recv_thread.start()
        self.is_receiving = True
        return True

    def stop_receiving(self):
        """Stop receiving data"""
        self.is_receiving = False
        self.stop_event.set()
        if self.recv_thread:
            self.recv_thread.join()

    def _receive_worker(self, num_samples):
        """Worker thread for receiving IQ data"""
        if not self.usrp:
            return

        # Set up stream
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        metadata = uhd.types.RXMetadata()
        streamer = self.usrp.get_rx_stream(st_args)

        # Start stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        streamer.issue_stream_cmd(stream_cmd)

        recv_buffer = np.zeros((1, num_samples), dtype=np.complex64)

        while not self.stop_event.is_set():
            try:
                streamer.recv(recv_buffer, metadata)
                if not metadata.error_code:
                    # Emit the received data
                    self.data_ready.emit(recv_buffer[0].copy())
                time.sleep(0.001)  # Small delay

            except Exception as e:
                self.status_changed.emit(f"Receive error: {str(e)}")
                break

        # Stop stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)


class SignalProcessor(QObject):
    """Signal processing class for demodulation and analysis"""

    def __init__(self):
        super().__init__()
        self.fs = 1e6  # Sample rate
        self.spectrum_history = []
        self.max_history = 100

    def compute_spectrum(self, iq_data, nfft=1024):
        """Compute power spectral density"""
        freqs, psd = welch(iq_data, fs=self.fs, nperseg=nfft, 
                          scaling='density', return_onesided=False)
        psd_db = 10 * np.log10(psd + 1e-12)  # Avoid log(0)
        return fftshift(freqs), fftshift(psd_db)

    def detect_peaks(self, psd_db, threshold=-50, prominence=5):
        """Detect signal peaks in spectrum"""
        peaks, properties = find_peaks(psd_db, height=threshold, 
                                      prominence=prominence, distance=10)
        return peaks, properties

    def update_spectrogram(self, psd_db):
        """Update spectrogram data for waterfall display"""
        self.spectrum_history.append(psd_db.copy())
        if len(self.spectrum_history) > self.max_history:
            self.spectrum_history.pop(0)
        return np.array(self.spectrum_history)

    def demodulate_bpsk(self, iq_data):
        """BPSK demodulation"""
        # Simple coherent detection assuming perfect synchronization
        symbols = np.real(iq_data)  # Extract real part for BPSK
        # Threshold detection
        bits = (symbols > 0).astype(int)
        return symbols, bits

    def demodulate_qpsk(self, iq_data):
        """QPSK demodulation"""
        # Simple QPSK demodulation
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)

        # Decision regions
        symbols = []
        for i, q in zip(i_data, q_data):
            if i >= 0 and q >= 0:
                symbols.append(0)  # 00
            elif i < 0 and q >= 0:
                symbols.append(1)  # 01
            elif i < 0 and q < 0:
                symbols.append(2)  # 11
            else:
                symbols.append(3)  # 10

        return iq_data, np.array(symbols)

    def demodulate_oqpsk(self, iq_data):
        """OQPSK (Offset QPSK) demodulation"""
        # Offset QPSK - Q component is delayed by half symbol
        # Simplified implementation
        return self.demodulate_qpsk(iq_data)

    def demodulate_8psk(self, iq_data):
        """8PSK demodulation"""
        phases = np.angle(iq_data)
        # 8 phase regions
        phase_step = 2 * np.pi / 8
        symbols = np.round((phases + np.pi) / phase_step) % 8
        return iq_data, symbols.astype(int)

    def demodulate_8qam(self, iq_data):
        """8QAM demodulation"""
        # Simplified 8QAM constellation
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)

        # Simple threshold-based decision
        symbols = []
        for i, q in zip(i_data, q_data):
            # 8QAM has 8 constellation points
            if abs(i) > abs(q):
                if i > 0:
                    symbols.append(0 if q > 0 else 7)
                else:
                    symbols.append(3 if q > 0 else 4)
            else:
                if q > 0:
                    symbols.append(1 if i > 0 else 2)
                else:
                    symbols.append(6 if i > 0 else 5)

        return iq_data, np.array(symbols)

    def classify_modulation(self, iq_data):
        """Simple automatic modulation classification"""
        # Calculate some basic features
        magnitude = np.abs(iq_data)
        phase = np.angle(iq_data)

        # Feature: amplitude variance (AM vs PM indicator)
        amp_var = np.var(magnitude)

        # Feature: phase continuity
        phase_diff = np.diff(np.unwrap(phase))
        phase_var = np.var(phase_diff)

        # Simple classification based on features
        if amp_var < 0.1 and phase_var > 1.0:
            return "PSK-like"
        elif amp_var > 0.1 and phase_var > 1.0:
            return "QAM-like"
        elif amp_var > 0.1 and phase_var < 1.0:
            return "AM-like"
        else:
            return "Unknown"


class MainWindow(QMainWindow):
    """Main SDR application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced SDR Application - USRP N210/X310")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize components
        self.usrp_interface = USRPInterface()
        self.signal_processor = SignalProcessor()

        # Data storage
        self.current_iq_data = None
        self.recording = False
        self.record_file = None

        # Setup GUI
        self.setup_ui()
        self.setup_connections()

        # Setup timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(50)  # 20 FPS

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # Right panel - Plots
        plot_panel = self.create_plot_panel()  
        main_layout.addWidget(plot_panel, 3)

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # USRP Connection
        conn_group = QGroupBox("USRP Connection")
        conn_layout = QVBoxLayout(conn_group)

        self.device_edit = QLineEdit("type=usrp2")  # Default for N210
        conn_layout.addWidget(QLabel("Device Args:"))
        conn_layout.addWidget(self.device_edit)

        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)

        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)

        # Frequency Controls
        freq_group = QGroupBox("Frequency Control")
        freq_layout = QGridLayout(freq_group)

        freq_layout.addWidget(QLabel("Center Freq (MHz):"), 0, 0)
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(10, 6000)
        self.freq_spinbox.setValue(100)
        self.freq_spinbox.setSuffix(" MHz")
        freq_layout.addWidget(self.freq_spinbox, 0, 1)

        freq_layout.addWidget(QLabel("Sample Rate (MHz):"), 1, 0)
        self.rate_spinbox = QDoubleSpinBox()
        self.rate_spinbox.setRange(0.1, 25)
        self.rate_spinbox.setValue(1.0)
        self.rate_spinbox.setSuffix(" MHz")
        freq_layout.addWidget(self.rate_spinbox, 1, 1)

        freq_layout.addWidget(QLabel("RX Gain (dB):"), 2, 0)
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(0, 70)
        self.gain_spinbox.setValue(50)
        freq_layout.addWidget(self.gain_spinbox, 2, 1)

        # Scan Controls
        scan_group = QGroupBox("Spectrum Scan")
        scan_layout = QGridLayout(scan_group)

        scan_layout.addWidget(QLabel("Start Freq (MHz):"), 0, 0)
        self.scan_start = QDoubleSpinBox()
        self.scan_start.setRange(10, 6000)
        self.scan_start.setValue(88)
        scan_layout.addWidget(self.scan_start, 0, 1)

        scan_layout.addWidget(QLabel("End Freq (MHz):"), 1, 0)
        self.scan_end = QDoubleSpinBox()
        self.scan_end.setRange(10, 6000) 
        self.scan_end.setValue(108)
        scan_layout.addWidget(self.scan_end, 1, 1)

        self.scan_btn = QPushButton("Start Scan")
        scan_layout.addWidget(self.scan_btn, 2, 0, 1, 2)

        # Demodulation Controls
        demod_group = QGroupBox("Demodulation")
        demod_layout = QVBoxLayout(demod_group)

        self.demod_combo = QComboBox()
        self.demod_combo.addItems(["Auto Detect", "BPSK", "QPSK", "OQPSK", "8PSK", "8QAM"])
        demod_layout.addWidget(self.demod_combo)

        self.demod_btn = QPushButton("Demodulate")
        demod_layout.addWidget(self.demod_btn)

        # Recording Controls
        record_group = QGroupBox("IQ Recording")
        record_layout = QVBoxLayout(record_group)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        record_layout.addWidget(self.record_btn)

        self.filename_edit = QLineEdit("iq_data.bin")
        record_layout.addWidget(QLabel("Filename:"))
        record_layout.addWidget(self.filename_edit)

        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        # Add all groups to panel
        layout.addWidget(conn_group)
        layout.addWidget(freq_group)  
        layout.addWidget(scan_group)
        layout.addWidget(demod_group)
        layout.addWidget(record_group)
        layout.addWidget(status_group)
        layout.addStretch()

        return panel

    def create_plot_panel(self):
        """Create plotting panel"""
        panel = QTabWidget()

        # Spectrum Tab
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)

        self.spectrum_plot = PlotWidget(title="Real-time Spectrum")
        self.spectrum_plot.setLabel('left', 'Power (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (MHz)')
        self.spectrum_plot.showGrid(True, True)
        spectrum_layout.addWidget(self.spectrum_plot)

        panel.addTab(spectrum_tab, "Spectrum")

        # Waterfall Tab
        waterfall_tab = QWidget()
        waterfall_layout = QVBoxLayout(waterfall_tab)

        self.waterfall_plot = PlotWidget(title="Spectrogram (Waterfall)")
        self.waterfall_plot.setLabel('left', 'Time')
        self.waterfall_plot.setLabel('bottom', 'Frequency (MHz)')

        self.waterfall_img = ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)

        # Colorbar for waterfall
        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.waterfall_img)
        self.colorbar.item.gradient.loadPreset('viridis')

        waterfall_h_layout = QHBoxLayout()
        waterfall_h_layout.addWidget(self.waterfall_plot, 4)
        waterfall_h_layout.addWidget(self.colorbar, 1)
        waterfall_layout.addLayout(waterfall_h_layout)

        panel.addTab(waterfall_tab, "Waterfall")

        # Constellation Tab
        constellation_tab = QWidget()
        constellation_layout = QVBoxLayout(constellation_tab)

        self.constellation_plot = PlotWidget(title="Constellation Diagram")
        self.constellation_plot.setLabel('left', 'Quadrature (Q)')
        self.constellation_plot.setLabel('bottom', 'In-phase (I)')
        self.constellation_plot.showGrid(True, True)
        self.constellation_plot.setAspectLocked(True)
        constellation_layout.addWidget(self.constellation_plot)

        panel.addTab(constellation_tab, "Constellation")

        return panel

    def setup_connections(self):
        """Setup signal-slot connections"""
        # USRP Interface connections
        self.connect_btn.clicked.connect(self.connect_usrp)
        self.disconnect_btn.clicked.connect(self.disconnect_usrp)
        self.usrp_interface.status_changed.connect(self.update_status)
        self.usrp_interface.data_ready.connect(self.process_iq_data)

        # Control connections
        self.freq_spinbox.valueChanged.connect(self.update_frequency)
        self.rate_spinbox.valueChanged.connect(self.update_sample_rate)
        self.gain_spinbox.valueChanged.connect(self.update_gain)
        self.scan_btn.clicked.connect(self.start_scan)
        self.demod_btn.clicked.connect(self.demodulate_signal)
        self.record_btn.clicked.connect(self.toggle_recording)

    def connect_usrp(self):
        """Connect to USRP"""
        device_args = self.device_edit.text()
        if self.usrp_interface.connect_usrp(device_args):
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            # Start receiving
            self.usrp_interface.start_receiving(1024)

    def disconnect_usrp(self):
        """Disconnect from USRP"""
        self.usrp_interface.disconnect_usrp()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def update_frequency(self, freq_mhz):
        """Update center frequency"""
        freq_hz = freq_mhz * 1e6
        self.usrp_interface.set_frequency(freq_hz)

    def update_sample_rate(self, rate_mhz):
        """Update sample rate"""
        rate_hz = rate_mhz * 1e6
        self.usrp_interface.set_sample_rate(rate_hz)
        self.signal_processor.fs = rate_hz

    def update_gain(self, gain_db):
        """Update RX gain"""
        self.usrp_interface.set_gain(gain_db)

    def start_scan(self):
        """Start frequency scan"""
        self.update_status("Starting frequency scan...")
        # Implementation for frequency scanning

    def demodulate_signal(self):
        """Demodulate current signal"""
        if self.current_iq_data is None:
            return

        demod_type = self.demod_combo.currentText()

        try:
            if demod_type == "BPSK":
                symbols, bits = self.signal_processor.demodulate_bpsk(self.current_iq_data)
                self.plot_constellation(symbols)
            elif demod_type == "QPSK":
                symbols, bits = self.signal_processor.demodulate_qpsk(self.current_iq_data)
                self.plot_constellation(symbols)
            elif demod_type == "OQPSK":
                symbols, bits = self.signal_processor.demodulate_oqpsk(self.current_iq_data)
                self.plot_constellation(symbols)
            elif demod_type == "8PSK":
                symbols, bits = self.signal_processor.demodulate_8psk(self.current_iq_data)
                self.plot_constellation(symbols)
            elif demod_type == "8QAM":
                symbols, bits = self.signal_processor.demodulate_8qam(self.current_iq_data)
                self.plot_constellation(symbols)
            elif demod_type == "Auto Detect":
                mod_type = self.signal_processor.classify_modulation(self.current_iq_data)
                self.update_status(f"Detected modulation: {mod_type}")

        except Exception as e:
            self.update_status(f"Demodulation error: {str(e)}")

    def toggle_recording(self):
        """Toggle IQ data recording"""
        if self.record_btn.isChecked():
            filename = self.filename_edit.text()
            try:
                self.record_file = open(filename, 'wb')
                self.recording = True
                self.record_btn.setText("Stop Recording")
                self.update_status(f"Recording to {filename}")
            except Exception as e:
                self.update_status(f"Recording error: {str(e)}")
                self.record_btn.setChecked(False)
        else:
            self.recording = False
            if self.record_file:
                self.record_file.close()
                self.record_file = None
            self.record_btn.setText("Start Recording")
            self.update_status("Recording stopped")

    def process_iq_data(self, iq_data):
        """Process incoming IQ data"""
        self.current_iq_data = iq_data

        # Record data if recording
        if self.recording and self.record_file:
            # Save as interleaved I/Q float32
            interleaved = np.column_stack((np.real(iq_data), np.imag(iq_data))).flatten()
            self.record_file.write(interleaved.astype(np.float32).tobytes())

    def update_plots(self):
        """Update all plots"""
        if self.current_iq_data is None:
            return

        try:
            # Update spectrum plot
            freqs, psd_db = self.signal_processor.compute_spectrum(self.current_iq_data)
            freq_mhz = (freqs + self.usrp_interface.center_freq) / 1e6

            self.spectrum_plot.clear()
            self.spectrum_plot.plot(freq_mhz, psd_db, pen='y')

            # Detect and mark peaks
            peaks, _ = self.signal_processor.detect_peaks(psd_db)
            if len(peaks) > 0:
                self.spectrum_plot.plot(freq_mhz[peaks], psd_db[peaks], 
                                       pen=None, symbol='o', symbolBrush='r')

            # Update waterfall
            spectrogram = self.signal_processor.update_spectrogram(psd_db)
            if len(spectrogram) > 1:
                self.waterfall_img.setImage(spectrogram.T, autoLevels=False)

        except Exception as e:
            pass  # Ignore plotting errors

    def plot_constellation(self, symbols):
        """Plot constellation diagram"""
        self.constellation_plot.clear()
        if len(symbols) > 1000:
            # Subsample for performance
            idx = np.random.choice(len(symbols), 1000, replace=False)
            symbols = symbols[idx]

        i_data = np.real(symbols)
        q_data = np.imag(symbols)
        self.constellation_plot.plot(i_data, q_data, pen=None, symbol='o', 
                                   symbolSize=3, symbolBrush='g')

    def update_status(self, message):
        """Update status text"""
        self.status_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Keep only last 50 lines
        if self.status_text.document().lineCount() > 50:
            cursor = self.status_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor)
            cursor.removeSelectedText()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
