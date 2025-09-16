"""
Icon resources for RF Spectrum Analyzer
Provides programmatically generated icons for the application
"""

from typing import Dict, Tuple
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QFont
from PySide6.QtCore import Qt, QRect
from PySide6.QtWidgets import QApplication

class IconGenerator:
    """Generate icons programmatically"""
    
    def __init__(self, size: int = 32):
        self.size = size
        self.icon_cache = {}
    
    def get_icon(self, icon_name: str, color: str = "#5E81AC") -> QIcon:
        """Get icon by name, generate if not cached"""
        cache_key = f"{icon_name}_{color}_{self.size}"
        
        if cache_key not in self.icon_cache:
            pixmap = self.generate_icon(icon_name, color)
            self.icon_cache[cache_key] = QIcon(pixmap)
        
        return self.icon_cache[cache_key]
    
    def generate_icon(self, icon_name: str, color: str) -> QPixmap:
        """Generate icon pixmap"""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(color))
        pen.setWidth(2)
        painter.setPen(pen)
        
        brush = QBrush(QColor(color))
        painter.setBrush(brush)
        
        if icon_name == "spectrum":
            self._draw_spectrum_icon(painter)
        elif icon_name == "waterfall":
            self._draw_waterfall_icon(painter)
        elif icon_name == "settings":
            self._draw_settings_icon(painter)
        elif icon_name == "play":
            self._draw_play_icon(painter)
        elif icon_name == "pause":
            self._draw_pause_icon(painter)
        elif icon_name == "stop":
            self._draw_stop_icon(painter)
        elif icon_name == "save":
            self._draw_save_icon(painter)
        elif icon_name == "load":
            self._draw_load_icon(painter)
        elif icon_name == "export":
            self._draw_export_icon(painter)
        elif icon_name == "help":
            self._draw_help_icon(painter)
        elif icon_name == "info":
            self._draw_info_icon(painter)
        elif icon_name == "warning":
            self._draw_warning_icon(painter)
        elif icon_name == "error":
            self._draw_error_icon(painter)
        elif icon_name == "device":
            self._draw_device_icon(painter)
        elif icon_name == "frequency":
            self._draw_frequency_icon(painter)
        elif icon_name == "gain":
            self._draw_gain_icon(painter)
        elif icon_name == "filter":
            self._draw_filter_icon(painter)
        elif icon_name == "peak":
            self._draw_peak_icon(painter)
        elif icon_name == "zoom_in":
            self._draw_zoom_in_icon(painter)
        elif icon_name == "zoom_out":
            self._draw_zoom_out_icon(painter)
        elif icon_name == "reset":
            self._draw_reset_icon(painter)
        else:
            self._draw_default_icon(painter)
        
        painter.end()
        return pixmap
    
    def _draw_spectrum_icon(self, painter: QPainter):
        """Draw spectrum analyzer icon"""
        # Draw frequency bars of different heights
        bar_width = self.size // 8
        for i in range(7):
            x = i * bar_width + bar_width // 2
            height = (i + 1) * self.size // 10 if i < 4 else (7 - i) * self.size // 10
            y = self.size - height - 4
            painter.drawRect(x, y, bar_width - 1, height)
    
    def _draw_waterfall_icon(self, painter: QPainter):
        """Draw waterfall display icon"""
        # Draw horizontal lines with gradient-like effect
        colors = ["#FF0000", "#FF8000", "#FFFF00", "#80FF00", "#00FF00", "#0080FF", "#0000FF"]
        line_height = self.size // len(colors)
        
        for i, color in enumerate(colors):
            painter.setPen(QPen(QColor(color), 2))
            y = i * line_height + line_height // 2
            painter.drawLine(4, y, self.size - 4, y)
    
    def _draw_settings_icon(self, painter: QPainter):
        """Draw settings gear icon"""
        center = self.size // 2
        outer_radius = center - 4
        inner_radius = center - 10
        
        # Draw gear teeth
        import math
        teeth = 8
        for i in range(teeth):
            angle = 2 * math.pi * i / teeth
            x1 = center + outer_radius * math.cos(angle)
            y1 = center + outer_radius * math.sin(angle)
            x2 = center + (outer_radius - 3) * math.cos(angle)
            y2 = center + (outer_radius - 3) * math.sin(angle)
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw center circle
        painter.drawEllipse(center - inner_radius, center - inner_radius, 
                          2 * inner_radius, 2 * inner_radius)
    
    def _draw_play_icon(self, painter: QPainter):
        """Draw play triangle icon"""
        points = [
            (self.size // 4, self.size // 4),
            (self.size // 4, 3 * self.size // 4),
            (3 * self.size // 4, self.size // 2)
        ]
        from PySide6.QtCore import QPoint
        painter.drawPolygon([QPoint(x, y) for x, y in points])
    
    def _draw_pause_icon(self, painter: QPainter):
        """Draw pause icon"""
        bar_width = self.size // 6
        x1 = self.size // 3 - bar_width // 2
        x2 = 2 * self.size // 3 - bar_width // 2
        y = self.size // 4
        height = self.size // 2
        
        painter.fillRect(x1, y, bar_width, height, painter.brush())
        painter.fillRect(x2, y, bar_width, height, painter.brush())
    
    def _draw_stop_icon(self, painter: QPainter):
        """Draw stop square icon"""
        size = self.size // 2
        x = (self.size - size) // 2
        y = (self.size - size) // 2
        painter.fillRect(x, y, size, size, painter.brush())
    
    def _draw_save_icon(self, painter: QPainter):
        """Draw save/disk icon"""
        # Draw floppy disk outline
        painter.drawRect(4, 4, self.size - 8, self.size - 8)
        # Draw label area
        painter.fillRect(6, 6, self.size - 12, self.size // 3, painter.brush())
        # Draw write protect tab
        painter.drawRect(self.size - 8, 4, 4, 6)
    
    def _draw_load_icon(self, painter: QPainter):
        """Draw load/folder icon"""
        # Draw folder shape
        painter.drawRect(4, self.size // 3, self.size - 8, 2 * self.size // 3 - 4)
        painter.drawRect(4, self.size // 4, self.size // 2, self.size // 6)
    
    def _draw_export_icon(self, painter: QPainter):
        """Draw export/upload icon"""
        # Draw arrow pointing up
        center = self.size // 2
        painter.drawLine(center, 4, center, self.size - 8)
        # Arrow head
        painter.drawLine(center, 4, center - 4, 8)
        painter.drawLine(center, 4, center + 4, 8)
        # Base line
        painter.drawLine(4, self.size - 4, self.size - 4, self.size - 4)
    
    def _draw_help_icon(self, painter: QPainter):
        """Draw help/question mark icon"""
        font = QFont()
        font.setPointSize(self.size // 2)
        font.setBold(True)
        painter.setFont(font)
        
        rect = QRect(0, 0, self.size, self.size)
        painter.drawText(rect, Qt.AlignCenter, "?")
    
    def _draw_info_icon(self, painter: QPainter):
        """Draw info icon"""
        center = self.size // 2
        radius = center - 4
        
        # Draw circle
        painter.drawEllipse(center - radius, center - radius, 2 * radius, 2 * radius)
        
        # Draw "i"
        font = QFont()
        font.setPointSize(self.size // 3)
        font.setBold(True)
        painter.setFont(font)
        
        rect = QRect(0, 0, self.size, self.size)
        painter.drawText(rect, Qt.AlignCenter, "i")
    
    def _draw_warning_icon(self, painter: QPainter):
        """Draw warning triangle icon"""
        # Draw triangle
        points = [
            (self.size // 2, 4),
            (4, self.size - 4),
            (self.size - 4, self.size - 4)
        ]
        from PySide6.QtCore import QPoint
        painter.drawPolygon([QPoint(x, y) for x, y in points])
        
        # Draw exclamation mark
        painter.setPen(QPen(QColor("white"), 2))
        center = self.size // 2
        painter.drawLine(center, self.size // 3, center, 2 * self.size // 3)
        painter.drawPoint(center, 3 * self.size // 4)
    
    def _draw_error_icon(self, painter: QPainter):
        """Draw error X icon"""
        # Draw X
        margin = 6
        painter.drawLine(margin, margin, self.size - margin, self.size - margin)
        painter.drawLine(self.size - margin, margin, margin, self.size - margin)
    
    def _draw_device_icon(self, painter: QPainter):
        """Draw SDR device icon"""
        # Draw device body
        painter.drawRect(4, self.size // 3, self.size - 8, self.size // 3)
        # Draw antenna
        painter.drawLine(8, self.size // 3, 8, 4)
        painter.drawLine(8, 4, 12, 8)
        painter.drawLine(8, 4, 4, 8)
        # Draw USB connector
        painter.drawRect(self.size - 8, self.size // 2 - 2, 4, 4)
    
    def _draw_frequency_icon(self, painter: QPainter):
        """Draw frequency/sine wave icon"""
        import math
        points = []
        for i in range(self.size):
            x = i
            y = self.size // 2 + int(self.size // 4 * math.sin(2 * math.pi * i / (self.size // 2)))
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    
    def _draw_gain_icon(self, painter: QPainter):
        """Draw gain/amplifier icon"""
        # Draw triangle (amplifier symbol)
        points = [
            (4, 4),
            (4, self.size - 4),
            (self.size - 8, self.size // 2)
        ]
        from PySide6.QtCore import QPoint
        painter.drawPolygon([QPoint(x, y) for x, y in points])
        
        # Draw output line
        painter.drawLine(self.size - 8, self.size // 2, self.size - 4, self.size // 2)
    
    def _draw_filter_icon(self, painter: QPainter):
        """Draw filter icon"""
        # Draw filter shape (funnel)
        painter.drawLine(4, 4, self.size - 4, 4)
        painter.drawLine(4, 4, self.size // 3, self.size // 2)
        painter.drawLine(self.size - 4, 4, 2 * self.size // 3, self.size // 2)
        painter.drawLine(self.size // 3, self.size // 2, 2 * self.size // 3, self.size // 2)
        painter.drawLine(self.size // 3, self.size // 2, self.size // 3, self.size - 4)
        painter.drawLine(2 * self.size // 3, self.size // 2, 2 * self.size // 3, self.size - 4)
    
    def _draw_peak_icon(self, painter: QPainter):
        """Draw peak detection icon"""
        # Draw mountain peaks
        points = [
            (4, self.size - 4),
            (self.size // 4, self.size // 3),
            (self.size // 2, 2 * self.size // 3),
            (3 * self.size // 4, self.size // 4),
            (self.size - 4, self.size - 4)
        ]
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    
    def _draw_zoom_in_icon(self, painter: QPainter):
        """Draw zoom in icon"""
        center = self.size // 2 - 2
        radius = self.size // 3
        
        # Draw magnifying glass
        painter.drawEllipse(center - radius, center - radius, 2 * radius, 2 * radius)
        painter.drawLine(center + radius - 2, center + radius - 2, self.size - 4, self.size - 4)
        
        # Draw plus sign
        painter.drawLine(center - 4, center, center + 4, center)
        painter.drawLine(center, center - 4, center, center + 4)
    
    def _draw_zoom_out_icon(self, painter: QPainter):
        """Draw zoom out icon"""
        center = self.size // 2 - 2
        radius = self.size // 3
        
        # Draw magnifying glass
        painter.drawEllipse(center - radius, center - radius, 2 * radius, 2 * radius)
        painter.drawLine(center + radius - 2, center + radius - 2, self.size - 4, self.size - 4)
        
        # Draw minus sign
        painter.drawLine(center - 4, center, center + 4, center)
    
    def _draw_reset_icon(self, painter: QPainter):
        """Draw reset/refresh icon"""
        import math
        center = self.size // 2
        radius = center - 4
        
        # Draw circular arrow
        for angle in range(0, 270, 5):
            rad1 = math.radians(angle)
            rad2 = math.radians(angle + 5)
            x1 = center + radius * math.cos(rad1)
            y1 = center + radius * math.sin(rad1)
            x2 = center + radius * math.cos(rad2)
            y2 = center + radius * math.sin(rad2)
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw arrow head
        painter.drawLine(self.size - 8, 8, self.size - 4, 4)
        painter.drawLine(self.size - 8, 8, self.size - 4, 12)
    
    def _draw_default_icon(self, painter: QPainter):
        """Draw default placeholder icon"""
        painter.drawRect(4, 4, self.size - 8, self.size - 8)
        painter.drawLine(4, 4, self.size - 4, self.size - 4)
        painter.drawLine(self.size - 4, 4, 4, self.size - 4)

class IconManager:
    """Manages application icons"""
    
    def __init__(self):
        self.generator = IconGenerator()
        self.icons = {}
    
    def get_icon(self, name: str, color: str = None) -> QIcon:
        """Get icon by name"""
        if color is None:
            # Use theme-appropriate color
            app = QApplication.instance()
            if app and app.palette().color(app.palette().Text).lightnessF() > 0.5:
                color = "#1E1E1E"  # Dark color for light theme
            else:
                color = "#ECEFF4"  # Light color for dark theme
        
        return self.generator.get_icon(name, color)
    
    def get_status_icon(self, status: str) -> QIcon:
        """Get status-specific icon with appropriate color"""
        color_map = {
            'success': '#A3BE8C',
            'warning': '#EBCB8B',
            'error': '#BF616A',
            'info': '#88C0D0'
        }
        
        icon_map = {
            'success': 'info',
            'warning': 'warning',
            'error': 'error',
            'info': 'info'
        }
        
        icon_name = icon_map.get(status, 'info')
        color = color_map.get(status, '#88C0D0')
        
        return self.generator.get_icon(icon_name, color)

# Global icon manager instance
icon_manager = IconManager()

# Convenience functions
def get_icon(name: str, color: str = None) -> QIcon:
    """Get icon by name"""
    return icon_manager.get_icon(name, color)

def get_status_icon(status: str) -> QIcon:
    """Get status icon"""
    return icon_manager.get_status_icon(status)