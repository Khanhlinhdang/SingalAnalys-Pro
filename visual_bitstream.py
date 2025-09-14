
"""
Visual Bitstream Display Widget
Display bit stream as colored pixels with configurable layout
"""

import numpy as np
import time
from typing import List, Optional, Tuple
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSpinBox, QCheckBox, QSlider,
                               QFrame, QScrollArea, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
import pyqtgraph as pg


class VisualBitstreamWidget(QWidget):
    """Visual bitstream display with colored pixels"""

    # Signal emitted when display is updated
    bitstream_updated = Signal(int)  # Total bits displayed

    def __init__(self):
        super().__init__()
        self.setup_ui()

        # Bitstream data
        self._bit_buffer = []
        self.max_bits = 10000

    @property
    def bit_buffer(self):
        """Get bit buffer with safe access"""
        if not hasattr(self, '_bit_buffer'):
            self._bit_buffer = []
        return self._bit_buffer
    
    @bit_buffer.setter
    def bit_buffer(self, value):
        """Set bit buffer"""
        self._bit_buffer = value if value is not None else []

        # Display parameters
        self.bits_per_row = 32
        self.pixel_size = 8
        self.bit_colors = {
            0: QColor(0, 0, 0),      # Black for 0
            1: QColor(0, 255, 0)     # Green for 1
        }

        # Statistics
        self.total_bits_received = 0
        self.last_update_time = time.time()
        self.bit_rate = 0.0
        self.zero_count = 0
        self.one_count = 0

        # Auto-scroll
        self.auto_scroll_enabled = True
        self._paused = False

    @property
    def paused(self):
        """Get paused state with safe access"""
        if not hasattr(self, '_paused'):
            self._paused = False
        return self._paused
    
    @paused.setter
    def paused(self, value):
        """Set paused state"""
        self._paused = bool(value)

    def setup_ui(self):
        """Setup visual bitstream UI"""
        layout = QVBoxLayout(self)

        # Header with controls
        header_group = QGroupBox("Visual Bitstream Display")
        header_layout = QGridLayout(header_group)

        # Row 1: Basic info
        self.title_label = QLabel("Visual Bit Stream")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(self.title_label, 0, 0)

        self.bit_count_label = QLabel("Bits: 0")
        self.bit_count_label.setStyleSheet("font-weight: bold; color: #14a085;")
        header_layout.addWidget(self.bit_count_label, 0, 1)

        self.bit_rate_label = QLabel("Rate: 0 bps")
        header_layout.addWidget(self.bit_rate_label, 0, 2)

        # Row 2: Display controls
        header_layout.addWidget(QLabel("Bits per row:"), 1, 0)
        self.bits_per_row_spinbox = QSpinBox()
        self.bits_per_row_spinbox.setRange(8, 128)
        self.bits_per_row_spinbox.setValue(32)
        self.bits_per_row_spinbox.valueChanged.connect(self.update_bits_per_row)
        header_layout.addWidget(self.bits_per_row_spinbox, 1, 1)

        header_layout.addWidget(QLabel("Pixel size:"), 1, 2)
        self.pixel_size_spinbox = QSpinBox()
        self.pixel_size_spinbox.setRange(4, 20)
        self.pixel_size_spinbox.setValue(8)
        self.pixel_size_spinbox.valueChanged.connect(self.update_pixel_size)
        header_layout.addWidget(self.pixel_size_spinbox, 1, 3)

        # Row 3: Control buttons
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        header_layout.addWidget(self.pause_btn, 2, 0)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_display)
        header_layout.addWidget(self.clear_btn, 2, 1)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_display)
        header_layout.addWidget(self.save_btn, 2, 2)

        # Auto scroll checkbox
        self.auto_scroll_cb = QCheckBox("Auto Scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.stateChanged.connect(self.toggle_auto_scroll)
        header_layout.addWidget(self.auto_scroll_cb, 2, 3)

        layout.addWidget(header_group)

        # Bit color legend
        legend_layout = QHBoxLayout()

        # Legend for bit 0
        zero_label = QLabel("0:")
        legend_layout.addWidget(zero_label)

        zero_color_label = QLabel()
        zero_color_label.setStyleSheet("background-color: black; border: 1px solid white;")
        zero_color_label.setFixedSize(20, 15)
        legend_layout.addWidget(zero_color_label)

        legend_layout.addWidget(QLabel("  "))

        # Legend for bit 1
        one_label = QLabel("1:")
        legend_layout.addWidget(one_label)

        one_color_label = QLabel()
        one_color_label.setStyleSheet("background-color: green; border: 1px solid white;")
        one_color_label.setFixedSize(20, 15)
        legend_layout.addWidget(one_color_label)

        legend_layout.addStretch()

        # Statistics
        self.stats_label = QLabel("0s: 0 (0%) | 1s: 0 (0%)")
        self.stats_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        legend_layout.addWidget(self.stats_label)

        layout.addLayout(legend_layout)

        # Main display area with scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create display widget
        self.display_widget = QLabel()
        self.display_widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.display_widget.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        self.display_widget.setMinimumSize(800, 400)

        self.scroll_area.setWidget(self.display_widget)
        layout.addWidget(self.scroll_area)

        # Status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.update_rate_label = QLabel("Update rate: Manual")
        self.update_rate_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        status_layout.addWidget(self.update_rate_label)

        layout.addLayout(status_layout)

        # Initialize display
        self.update_display()

    def add_bits(self, new_bits):
        """Add new bits to the display"""
        if self.paused or new_bits is None:
            return

        try:
            # Convert to list of integers
            if isinstance(new_bits, np.ndarray):
                bit_list = new_bits.astype(int).tolist()
            else:
                bit_list = [int(b) for b in new_bits]

            # Add to buffer
            self.bit_buffer.extend(bit_list)

            # Limit buffer size
            if len(self.bit_buffer) > self.max_bits:
                removed_bits = self.bit_buffer[:-self.max_bits]
                self.bit_buffer = self.bit_buffer[-self.max_bits:]

                # Update counts for removed bits
                for bit in removed_bits:
                    if bit == 0:
                        self.zero_count = max(0, self.zero_count - 1)
                    else:
                        self.one_count = max(0, self.one_count - 1)

            # Update statistics
            self.total_bits_received += len(bit_list)

            # Count new bits
            for bit in bit_list:
                if bit == 0:
                    self.zero_count += 1
                else:
                    self.one_count += 1

            # Update bit rate
            current_time = time.time()
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                self.bit_rate = len(bit_list) / time_diff
            self.last_update_time = current_time

            # Update display
            self.update_display()

            # Update labels
            self.update_labels()

            # Auto scroll to bottom
            if self.auto_scroll_enabled:
                self.scroll_to_bottom()

            # Emit signal
            self.bitstream_updated.emit(len(self.bit_buffer))

        except Exception as e:
            print(f"Error adding bits: {e}")
            self.status_label.setText(f"Error: {str(e)}")

    def update_display(self):
        """Update the visual display"""
        try:
            if not self.bit_buffer:
                # Empty display
                pixmap = QPixmap(800, 100)
                pixmap.fill(QColor(42, 42, 42))  # Dark gray background
                self.display_widget.setPixmap(pixmap)
                return

            # Calculate display dimensions
            num_rows = (len(self.bit_buffer) + self.bits_per_row - 1) // self.bits_per_row
            display_width = self.bits_per_row * self.pixel_size
            display_height = num_rows * self.pixel_size

            # Create pixmap
            pixmap = QPixmap(display_width, display_height)
            pixmap.fill(QColor(42, 42, 42))  # Background color

            # Draw bits
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, False)  # Pixel perfect

            for i, bit in enumerate(self.bit_buffer):
                row = i // self.bits_per_row
                col = i % self.bits_per_row

                x = col * self.pixel_size
                y = row * self.pixel_size

                color = self.bit_colors.get(bit, QColor(128, 128, 128))  # Gray for unknown
                painter.fillRect(x, y, self.pixel_size, self.pixel_size, color)

            painter.end()

            # Set pixmap to label
            self.display_widget.setPixmap(pixmap)
            self.display_widget.setMinimumSize(display_width, max(display_height, 100))

        except Exception as e:
            print(f"Display update error: {e}")
            self.status_label.setText(f"Display error: {str(e)}")

    def update_labels(self):
        """Update information labels"""
        # Bit count
        self.bit_count_label.setText(f"Bits: {len(self.bit_buffer)}")

        # Bit rate
        self.bit_rate_label.setText(f"Rate: {self.bit_rate:.1f} bps")

        # Statistics
        total_bits = len(self.bit_buffer)
        if total_bits > 0:
            zero_percent = (self.zero_count / total_bits) * 100
            one_percent = (self.one_count / total_bits) * 100
            self.stats_label.setText(f"0s: {self.zero_count} ({zero_percent:.1f}%) | 1s: {self.one_count} ({one_percent:.1f}%)")
        else:
            self.stats_label.setText("0s: 0 (0%) | 1s: 0 (0%)")

        # Status
        if self.paused:
            self.status_label.setText("Paused")
        elif len(self.bit_buffer) > 0:
            self.status_label.setText(f"Displaying {len(self.bit_buffer)} bits")
        else:
            self.status_label.setText("Ready")

    def update_bits_per_row(self, value):
        """Update bits per row setting"""
        self.bits_per_row = value
        self.update_display()
        self.status_label.setText(f"Changed to {value} bits per row")

    def update_pixel_size(self, value):
        """Update pixel size setting"""
        self.pixel_size = value
        self.update_display()
        self.status_label.setText(f"Changed to {value}x{value} pixel size")

    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = self.pause_btn.isChecked()
        if self.paused:
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused")
        else:
            self.pause_btn.setText("Pause")
            self.status_label.setText("Running")

    def toggle_auto_scroll(self, state):
        """Toggle auto scroll"""
        self.auto_scroll_enabled = state == Qt.Checked

    def scroll_to_bottom(self):
        """Scroll to bottom of display"""
        try:
            v_scrollbar = self.scroll_area.verticalScrollBar()
            v_scrollbar.setValue(v_scrollbar.maximum())
        except:
            pass

    def clear_display(self):
        """Clear the display"""
        self.bit_buffer.clear()
        self.total_bits_received = 0
        self.zero_count = 0
        self.one_count = 0
        self.bit_rate = 0.0

        self.update_display()
        self.update_labels()
        self.status_label.setText("Display cleared")

    def save_display(self):
        """Save display as image"""
        try:
            pixmap = self.display_widget.pixmap()
            if pixmap:
                filename = f"bitstream_{int(time.time())}.png"
                success = pixmap.save(filename)
                if success:
                    self.status_label.setText(f"Saved as {filename}")
                else:
                    self.status_label.setText("Failed to save image")
            else:
                self.status_label.setText("No image to save")
        except Exception as e:
            self.status_label.setText(f"Save error: {str(e)}")

    def get_display_info(self):
        """Get current display information"""
        return {
            'total_bits': len(self.bit_buffer),
            'bits_per_row': self.bits_per_row,
            'pixel_size': self.pixel_size,
            'bit_rate': self.bit_rate,
            'zero_count': self.zero_count,
            'one_count': self.one_count,
            'paused': self.paused
        }

    def set_bit_colors(self, zero_color, one_color):
        """Set custom bit colors"""
        self.bit_colors[0] = QColor(zero_color) if isinstance(zero_color, str) else zero_color
        self.bit_colors[1] = QColor(one_color) if isinstance(one_color, str) else one_color
        self.update_display()

    def export_bits(self, format_type='binary'):
        """Export bits in various formats"""
        if not self.bit_buffer:
            return ""

        try:
            if format_type == 'binary':
                return ''.join(map(str, self.bit_buffer))
            elif format_type == 'hex':
                # Group into bytes and convert to hex
                hex_chars = []
                for i in range(0, len(self.bit_buffer), 8):
                    byte_bits = self.bit_buffer[i:i+8]
                    # Pad if incomplete byte
                    while len(byte_bits) < 8:
                        byte_bits.append(0)

                    byte_value = 0
                    for j, bit in enumerate(byte_bits):
                        byte_value |= (bit << (7-j))

                    hex_chars.append(f"{byte_value:02X}")

                return ' '.join(hex_chars)
            elif format_type == 'decimal':
                # Convert to decimal string
                return str(int(''.join(map(str, self.bit_buffer)), 2))
            else:
                return ''.join(map(str, self.bit_buffer))

        except Exception as e:
            print(f"Export error: {e}")
            return ""


class BitstreamAnalyzer(QWidget):
    """Additional bitstream analysis widget"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.bit_history = []
        self.analysis_results = {}

    def setup_ui(self):
        """Setup analyzer UI"""
        layout = QVBoxLayout(self)

        # Analysis controls
        controls_group = QGroupBox("Bitstream Analysis")
        controls_layout = QGridLayout(controls_group)

        self.analyze_btn = QPushButton("Analyze Pattern")
        self.analyze_btn.clicked.connect(self.analyze_bits)
        controls_layout.addWidget(self.analyze_btn, 0, 0)

        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_analysis)
        controls_layout.addWidget(self.export_btn, 0, 1)

        layout.addWidget(controls_group)

        # Analysis results
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QLabel("No analysis performed")
        self.results_text.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.results_text.setWordWrap(True)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Pattern statistics plot
        self.pattern_plot = pg.PlotWidget(title="Bit Pattern Statistics")
        self.pattern_plot.setLabel('left', 'Count')
        self.pattern_plot.setLabel('bottom', 'Pattern Length')
        self.pattern_plot.showGrid(True, True)
        layout.addWidget(self.pattern_plot)

    def update_bits(self, bit_buffer):
        """Update with new bit data"""
        self.bit_history = bit_buffer.copy()

    def analyze_bits(self):
        """Analyze bit patterns"""
        if not self.bit_history:
            self.results_text.setText("No bit data to analyze")
            return

        try:
            # Basic statistics
            total_bits = len(self.bit_history)
            zero_count = self.bit_history.count(0)
            one_count = self.bit_history.count(1)

            # Pattern analysis
            patterns_2bit = self._count_patterns(2)
            patterns_3bit = self._count_patterns(3)
            patterns_4bit = self._count_patterns(4)

            # Transition analysis
            transitions = self._count_transitions()

            # Entropy estimation
            entropy = self._estimate_entropy()

            # Format results
            results = []
            results.append(f"=== BITSTREAM ANALYSIS ===")
            results.append(f"Total bits: {total_bits}")
            results.append(f"Zeros: {zero_count} ({zero_count/total_bits*100:.1f}%)")
            results.append(f"Ones: {one_count} ({one_count/total_bits*100:.1f}%)")
            results.append(f"")
            results.append(f"Transitions (0→1, 1→0): {transitions}")
            results.append(f"Estimated entropy: {entropy:.3f} bits/symbol")
            results.append(f"")
            results.append(f"2-bit patterns: {patterns_2bit}")
            results.append(f"3-bit patterns: {patterns_3bit}")
            results.append(f"4-bit patterns: {patterns_4bit}")

            self.results_text.setText("\n".join(results))

            # Update plot
            self._update_pattern_plot()

        except Exception as e:
            self.results_text.setText(f"Analysis error: {str(e)}")

    def _count_patterns(self, length):
        """Count bit patterns of given length"""
        if len(self.bit_history) < length:
            return {}

        patterns = {}
        for i in range(len(self.bit_history) - length + 1):
            pattern = tuple(self.bit_history[i:i+length])
            patterns[pattern] = patterns.get(pattern, 0) + 1

        return patterns

    def _count_transitions(self):
        """Count bit transitions"""
        if len(self.bit_history) < 2:
            return 0

        transitions = 0
        for i in range(len(self.bit_history) - 1):
            if self.bit_history[i] != self.bit_history[i+1]:
                transitions += 1

        return transitions

    def _estimate_entropy(self):
        """Estimate entropy of bit sequence"""
        if not self.bit_history:
            return 0

        total = len(self.bit_history)
        zero_count = self.bit_history.count(0)
        one_count = self.bit_history.count(1)

        if zero_count == 0 or one_count == 0:
            return 0  # No entropy if all bits are same

        p0 = zero_count / total
        p1 = one_count / total

        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        return entropy

    def _update_pattern_plot(self):
        """Update pattern statistics plot"""
        try:
            self.pattern_plot.clear()

            # Plot 2-bit, 3-bit, 4-bit pattern counts
            pattern_lengths = [2, 3, 4]
            colors = ['r', 'g', 'b']

            for length, color in zip(pattern_lengths, colors):
                patterns = self._count_patterns(length)
                if patterns:
                    x_data = list(range(len(patterns)))
                    y_data = list(patterns.values())

                    self.pattern_plot.plot(x_data, y_data, pen=color, 
                                         symbol='o', symbolBrush=color,
                                         name=f"{length}-bit patterns")
        except Exception as e:
            print(f"Plot update error: {e}")

    def export_analysis(self):
        """Export analysis results"""
        if not self.analysis_results:
            self.analyze_bits()  # Run analysis first

        try:
            filename = f"bitstream_analysis_{int(time.time())}.txt"
            with open(filename, 'w') as f:
                f.write(self.results_text.text())

            self.results_text.setText(self.results_text.text() + f"\n\nAnalysis exported to {filename}")

        except Exception as e:
            self.results_text.setText(self.results_text.text() + f"\n\nExport error: {str(e)}")


# Test visual bitstream display
def test_visual_bitstream():
    """Test visual bitstream widget"""
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Create test window
    widget = VisualBitstreamWidget()
    widget.show()
    widget.setWindowTitle("Visual Bitstream Test")

    # Generate test bit pattern
    test_bits = []

    # Add various patterns
    test_bits.extend([1, 0, 1, 0, 1, 0, 1, 0])  # Alternating
    test_bits.extend([1, 1, 1, 1, 0, 0, 0, 0])  # Blocks
    test_bits.extend([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])  # Random pattern

    # Repeat pattern
    test_bits = test_bits * 20

    # Add test bits
    widget.add_bits(test_bits)

    print("✅ Visual bitstream widget test started")
    print("  - Green pixels = 1 bits")
    print("  - Black pixels = 0 bits")
    print("  - Adjust bits per row and pixel size")

    sys.exit(app.exec())


if __name__ == "__main__":
    test_visual_bitstream()
