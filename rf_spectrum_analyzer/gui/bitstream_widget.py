"""
Bitstream Display Widget for RF Spectrum Analyzer
Visual representation of bit streams with colored pixels and comprehensive analysis.
"""

import sys
import time
import numpy as np
from typing import List, Optional, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSpinBox, QCheckBox, QSlider,
    QFrame, QScrollArea, QGroupBox, QGridLayout,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
import pyqtgraph as pg


class BitstreamWidget(QWidget):
    """Bitstream display widget with visual representation and analysis."""

    # Signals
    bitstream_updated = Signal(int)  # Total bits displayed
    analysis_completed = Signal(dict)  # Analysis results

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_data()
        
    def setup_data(self):
        """Initialize data structures."""
        # Bitstream data
        self._bit_buffer = []
        self.max_bits = 50000  # Increased for RF applications
        
        # Display parameters
        self.bits_per_row = 32
        self.pixel_size = 6  # Smaller for more data density
        self.bit_colors = {
            0: QColor(20, 20, 20),      # Dark gray for 0
            1: QColor(0, 255, 100)      # Bright green for 1
        }
        
        # Statistics
        self.total_bits_received = 0
        self.last_update_time = time.time()
        self.bit_rate = 0.0
        self.zero_count = 0
        self.one_count = 0
        
        # State
        self.auto_scroll_enabled = True
        self._paused = False
        
        # Performance optimization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.pending_bits = []

    @property
    def bit_buffer(self):
        """Get bit buffer with safe access."""
        if not hasattr(self, '_bit_buffer'):
            self._bit_buffer = []
        return self._bit_buffer
    
    @bit_buffer.setter
    def bit_buffer(self, value):
        """Set bit buffer."""
        self._bit_buffer = value if value is not None else []

    @property
    def paused(self):
        """Get paused state with safe access."""
        if not hasattr(self, '_paused'):
            self._paused = False
        return self._paused
    
    @paused.setter
    def paused(self, value):
        """Set paused state."""
        self._paused = bool(value)

    def setup_ui(self):
        """Setup bitstream widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Header with controls
        header_group = QGroupBox("Bitstream Visualization")
        header_group.setStyleSheet("QGroupBox { font-weight: bold; color: #14a085; }")
        header_layout = QGridLayout(header_group)

        # Row 1: Status information
        self.title_label = QLabel("RF Bitstream Display")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #ffffff;")
        header_layout.addWidget(self.title_label, 0, 0, 1, 2)

        self.bit_count_label = QLabel("Bits: 0")
        self.bit_count_label.setStyleSheet("font-weight: bold; color: #14a085;")
        header_layout.addWidget(self.bit_count_label, 0, 2)

        self.bit_rate_label = QLabel("Rate: 0 bps")
        self.bit_rate_label.setStyleSheet("color: #cccccc;")
        header_layout.addWidget(self.bit_rate_label, 0, 3)

        # Row 2: Display configuration
        header_layout.addWidget(QLabel("Bits/row:"), 1, 0)
        self.bits_per_row_spinbox = QSpinBox()
        self.bits_per_row_spinbox.setRange(16, 128)
        self.bits_per_row_spinbox.setValue(32)
        self.bits_per_row_spinbox.valueChanged.connect(self.update_bits_per_row)
        header_layout.addWidget(self.bits_per_row_spinbox, 1, 1)

        header_layout.addWidget(QLabel("Pixel size:"), 1, 2)
        self.pixel_size_spinbox = QSpinBox()
        self.pixel_size_spinbox.setRange(3, 15)
        self.pixel_size_spinbox.setValue(6)
        self.pixel_size_spinbox.valueChanged.connect(self.update_pixel_size)
        header_layout.addWidget(self.pixel_size_spinbox, 1, 3)

        # Row 3: Control buttons
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setStyleSheet("""
            QPushButton { background-color: #2a2a2a; border: 1px solid #555; padding: 5px; }
            QPushButton:checked { background-color: #d32f2f; }
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)
        header_layout.addWidget(self.pause_btn, 2, 0)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("QPushButton { background-color: #2a2a2a; border: 1px solid #555; padding: 5px; }")
        self.clear_btn.clicked.connect(self.clear_display)
        header_layout.addWidget(self.clear_btn, 2, 1)

        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet("QPushButton { background-color: #2a2a2a; border: 1px solid #555; padding: 5px; }")
        self.save_btn.clicked.connect(self.save_display)
        header_layout.addWidget(self.save_btn, 2, 2)

        # Auto scroll checkbox
        self.auto_scroll_cb = QCheckBox("Auto Scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setStyleSheet("QCheckBox { color: #cccccc; }")
        self.auto_scroll_cb.stateChanged.connect(self.toggle_auto_scroll)
        header_layout.addWidget(self.auto_scroll_cb, 2, 3)

        layout.addWidget(header_group)

        # Legend and statistics
        info_layout = QHBoxLayout()

        # Bit color legend
        legend_frame = QFrame()
        legend_frame.setFrameStyle(QFrame.StyledPanel)
        legend_layout = QHBoxLayout(legend_frame)

        legend_layout.addWidget(QLabel("Legend:"))
        
        # 0 bit legend
        zero_label = QLabel("0:")
        zero_label.setStyleSheet("color: #cccccc;")
        legend_layout.addWidget(zero_label)

        zero_color_label = QLabel()
        zero_color_label.setStyleSheet("background-color: #141414; border: 1px solid #555;")
        zero_color_label.setFixedSize(20, 15)
        legend_layout.addWidget(zero_color_label)

        legend_layout.addWidget(QLabel("  "))

        # 1 bit legend
        one_label = QLabel("1:")
        one_label.setStyleSheet("color: #cccccc;")
        legend_layout.addWidget(one_label)

        one_color_label = QLabel()
        one_color_label.setStyleSheet("background-color: #00ff64; border: 1px solid #555;")
        one_color_label.setFixedSize(20, 15)
        legend_layout.addWidget(one_color_label)

        legend_layout.addStretch()
        info_layout.addWidget(legend_frame)

        # Statistics display
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)

        self.stats_label = QLabel("0s: 0 (0%) | 1s: 0 (0%)")
        self.stats_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        stats_layout.addWidget(self.stats_label)

        self.entropy_label = QLabel("Entropy: 0.00 bits")
        self.entropy_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        stats_layout.addWidget(self.entropy_label)

        info_layout.addWidget(stats_frame)
        layout.addLayout(info_layout)

        # Main display area with scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea { 
                background-color: #1a1a1a; 
                border: 2px solid #14a085; 
                border-radius: 5px; 
            }
        """)

        # Create display widget
        self.display_widget = QLabel()
        self.display_widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.display_widget.setStyleSheet("background-color: #1a1a1a; border: none;")
        self.display_widget.setMinimumSize(800, 300)

        self.scroll_area.setWidget(self.display_widget)
        layout.addWidget(self.scroll_area)

        # Status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready for bitstream data")
        self.status_label.setStyleSheet("color: #14a085; font-size: 9pt; font-weight: bold;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.buffer_usage_label = QLabel("Buffer: 0%")
        self.buffer_usage_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        status_layout.addWidget(self.buffer_usage_label)

        layout.addLayout(status_layout)

        # Initialize display
        self.update_display()

    def add_bits(self, new_bits):
        """Add new bits to the display buffer."""
        if self.paused or new_bits is None:
            return

        try:
            # Convert to list of integers
            if isinstance(new_bits, np.ndarray):
                bit_list = new_bits.astype(int).tolist()
            elif isinstance(new_bits, (list, tuple)):
                bit_list = [int(b) for b in new_bits]
            else:
                # Single bit
                bit_list = [int(new_bits)]

            # Add to buffer
            self.bit_buffer.extend(bit_list)

            # Limit buffer size
            if len(self.bit_buffer) > self.max_bits:
                # Remove oldest bits
                excess = len(self.bit_buffer) - self.max_bits
                self.bit_buffer = self.bit_buffer[excess:]
                
                # Adjust counts
                removed_bits = bit_list[:excess] if excess <= len(bit_list) else []
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

            # Batch updates for performance
            if not self.update_timer.isActive():
                self.update_timer.start(50)  # Update every 50ms

            # Update labels immediately for responsiveness
            self.update_labels()

            # Auto scroll
            if self.auto_scroll_enabled:
                self.scroll_to_bottom()

            # Emit signal
            self.bitstream_updated.emit(len(self.bit_buffer))

        except Exception as e:
            print(f"Error adding bits: {e}")
            self.status_label.setText(f"Error: {str(e)}")

    def update_display(self):
        """Update the visual bitstream display."""
        try:
            if not self.bit_buffer:
                # Create empty display
                empty_pixmap = QPixmap(400, 100)
                empty_pixmap.fill(QColor(26, 26, 26))
                painter = QPainter(empty_pixmap)
                painter.setPen(QColor(204, 204, 204))
                painter.drawText(empty_pixmap.rect(), Qt.AlignCenter, "No bitstream data")
                painter.end()
                self.display_widget.setPixmap(empty_pixmap)
                return

            # Calculate display dimensions
            num_rows = (len(self.bit_buffer) + self.bits_per_row - 1) // self.bits_per_row
            display_width = self.bits_per_row * self.pixel_size
            display_height = max(num_rows * self.pixel_size, 100)

            # Create pixmap
            pixmap = QPixmap(display_width, display_height)
            pixmap.fill(QColor(26, 26, 26))  # Background color

            # Draw bits
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, False)  # Pixel perfect

            for i, bit in enumerate(self.bit_buffer):
                row = i // self.bits_per_row
                col = i % self.bits_per_row
                
                x = col * self.pixel_size
                y = row * self.pixel_size
                
                color = self.bit_colors.get(bit, QColor(255, 0, 0))  # Red for invalid bits
                painter.fillRect(x, y, self.pixel_size - 1, self.pixel_size - 1, color)

            painter.end()

            # Set pixmap to label
            self.display_widget.setPixmap(pixmap)
            self.display_widget.setMinimumSize(display_width, display_height)

            # Stop update timer
            self.update_timer.stop()

        except Exception as e:
            print(f"Display update error: {e}")
            self.status_label.setText(f"Display error: {str(e)}")

    def update_labels(self):
        """Update information labels."""
        # Bit count
        self.bit_count_label.setText(f"Bits: {len(self.bit_buffer):,}")

        # Bit rate
        if self.bit_rate > 1000:
            rate_text = f"Rate: {self.bit_rate/1000:.1f} kbps"
        else:
            rate_text = f"Rate: {self.bit_rate:.1f} bps"
        self.bit_rate_label.setText(rate_text)

        # Statistics
        total_bits = len(self.bit_buffer)
        if total_bits > 0:
            zero_percent = (self.zero_count / total_bits) * 100
            one_percent = (self.one_count / total_bits) * 100
            self.stats_label.setText(f"0s: {self.zero_count:,} ({zero_percent:.1f}%) | 1s: {self.one_count:,} ({one_percent:.1f}%)")
            
            # Calculate entropy
            if zero_percent > 0 and one_percent > 0:
                p0 = zero_percent / 100
                p1 = one_percent / 100
                entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                self.entropy_label.setText(f"Entropy: {entropy:.3f} bits")
            else:
                self.entropy_label.setText("Entropy: 0.000 bits")
        else:
            self.stats_label.setText("0s: 0 (0%) | 1s: 0 (0%)")
            self.entropy_label.setText("Entropy: 0.000 bits")

        # Buffer usage
        buffer_percent = (len(self.bit_buffer) / self.max_bits) * 100
        self.buffer_usage_label.setText(f"Buffer: {buffer_percent:.1f}%")

        # Status
        if self.paused:
            self.status_label.setText("Paused - Data acquisition stopped")
        elif len(self.bit_buffer) > 0:
            self.status_label.setText(f"Active - Displaying {len(self.bit_buffer):,} bits")
        else:
            self.status_label.setText("Ready for bitstream data")

    def update_bits_per_row(self, value):
        """Update bits per row setting."""
        self.bits_per_row = value
        self.update_display()
        self.status_label.setText(f"Layout: {value} bits per row")

    def update_pixel_size(self, value):
        """Update pixel size setting."""
        self.pixel_size = value
        self.update_display()
        self.status_label.setText(f"Display: {value}×{value} pixel size")

    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = self.pause_btn.isChecked()
        if self.paused:
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused - Data acquisition stopped")
        else:
            self.pause_btn.setText("Pause")
            self.status_label.setText("Active - Ready for data")

    def toggle_auto_scroll(self, state):
        """Toggle auto scroll functionality."""
        self.auto_scroll_enabled = state == Qt.Checked
        status = "enabled" if self.auto_scroll_enabled else "disabled"
        self.status_label.setText(f"Auto-scroll {status}")

    def scroll_to_bottom(self):
        """Scroll to bottom of display."""
        try:
            v_scrollbar = self.scroll_area.verticalScrollBar()
            v_scrollbar.setValue(v_scrollbar.maximum())
        except:
            pass

    def clear_display(self):
        """Clear the bitstream display."""
        self.bit_buffer.clear()
        self.total_bits_received = 0
        self.zero_count = 0
        self.one_count = 0
        self.bit_rate = 0.0

        self.update_display()
        self.update_labels()
        self.status_label.setText("Display cleared - Ready for new data")

    def save_display(self):
        """Save display as image."""
        try:
            pixmap = self.display_widget.pixmap()
            if pixmap and not pixmap.isNull():
                filename, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Save Bitstream Display", 
                    f"bitstream_{int(time.time())}.png",
                    "PNG files (*.png);;All files (*.*)"
                )
                if filename:
                    pixmap.save(filename)
                    self.status_label.setText(f"Saved: {filename}")
            else:
                QMessageBox.warning(self, "Warning", "No data to save")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save error: {str(e)}")
            self.status_label.setText(f"Save error: {str(e)}")

    def get_display_info(self):
        """Get current display information."""
        return {
            'total_bits': len(self.bit_buffer),
            'bits_per_row': self.bits_per_row,
            'pixel_size': self.pixel_size,
            'bit_rate': self.bit_rate,
            'zero_count': self.zero_count,
            'one_count': self.one_count,
            'paused': self.paused,
            'entropy': self.calculate_entropy(),
            'buffer_usage': len(self.bit_buffer) / self.max_bits
        }

    def calculate_entropy(self):
        """Calculate Shannon entropy of current bit buffer."""
        if not self.bit_buffer:
            return 0.0
        
        total = len(self.bit_buffer)
        if self.zero_count == 0 or self.one_count == 0:
            return 0.0
        
        p0 = self.zero_count / total
        p1 = self.one_count / total
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    def set_bit_colors(self, zero_color, one_color):
        """Set custom bit colors."""
        self.bit_colors[0] = QColor(zero_color) if isinstance(zero_color, str) else zero_color
        self.bit_colors[1] = QColor(one_color) if isinstance(one_color, str) else one_color
        self.update_display()
        self.status_label.setText("Color scheme updated")

    def export_bits(self, format_type='binary'):
        """Export bits in various formats."""
        if not self.bit_buffer:
            return ""

        try:
            if format_type == 'binary':
                return ''.join(map(str, self.bit_buffer))
            elif format_type == 'hex':
                # Convert bits to hex
                bit_string = ''.join(map(str, self.bit_buffer))
                # Pad to multiple of 4
                while len(bit_string) % 4 != 0:
                    bit_string = '0' + bit_string
                
                hex_result = ''
                for i in range(0, len(bit_string), 4):
                    nibble = bit_string[i:i+4]
                    hex_result += format(int(nibble, 2), 'X')
                return hex_result
            elif format_type == 'numpy':
                return np.array(self.bit_buffer, dtype=np.uint8)
            else:
                return ''.join(map(str, self.bit_buffer))

        except Exception as e:
            print(f"Export error: {e}")
            return ""