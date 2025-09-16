"""
Theme management for RF Spectrum Analyzer
Provides dark and light theme definitions for the application
"""

from typing import Dict, Any
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

@dataclass
class ThemeColors:
    """Color scheme for a theme"""
    # Background colors
    background: str
    background_alternate: str
    background_selected: str
    
    # Text colors
    text: str
    text_disabled: str
    text_selected: str
    
    # Border and separator colors
    border: str
    separator: str
    
    # Accent colors
    accent: str
    accent_light: str
    accent_dark: str
    
    # Status colors
    success: str
    warning: str
    error: str
    info: str
    
    # Plot colors
    plot_background: str
    plot_grid: str
    plot_axis: str
    plot_line: str

# Dark theme color scheme
DARK_THEME = ThemeColors(
    background="#2E3440",
    background_alternate="#3B4252",
    background_selected="#5E81AC",
    text="#ECEFF4",
    text_disabled="#6C7B95",
    text_selected="#FFFFFF",
    border="#4C566A",
    separator="#434C5E",
    accent="#5E81AC",
    accent_light="#81A1C1",
    accent_dark="#4A6A94",
    success="#A3BE8C",
    warning="#EBCB8B",
    error="#BF616A",
    info="#88C0D0",
    plot_background="#2E3440",
    plot_grid="#434C5E",
    plot_axis="#D8DEE9",
    plot_line="#88C0D0"
)

# Light theme color scheme
LIGHT_THEME = ThemeColors(
    background="#FFFFFF",
    background_alternate="#F8F9FA",
    background_selected="#007ACC",
    text="#1E1E1E",
    text_disabled="#6C6C6C",
    text_selected="#FFFFFF",
    border="#E1E1E1",
    separator="#D4D4D4",
    accent="#007ACC",
    accent_light="#4A9EF1",
    accent_dark="#005A9E",
    success="#107C10",
    warning="#FF8C00",
    error="#E74C3C",
    info="#0078D4",
    plot_background="#FFFFFF",
    plot_grid="#E5E5E5",
    plot_axis="#333333",
    plot_line="#007ACC"
)

class ThemeManager(QObject):
    """Manages application themes"""
    
    theme_changed = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.current_theme = "dark"
        self.themes = {
            "dark": DARK_THEME,
            "light": LIGHT_THEME
        }
    
    def get_theme_colors(self, theme_name: str = None) -> ThemeColors:
        """Get color scheme for specified theme"""
        if theme_name is None:
            theme_name = self.current_theme
        
        return self.themes.get(theme_name, DARK_THEME)
    
    def set_theme(self, theme_name: str):
        """Set the current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            self.apply_theme()
            self.theme_changed.emit(theme_name)
    
    def apply_theme(self):
        """Apply current theme to the application"""
        app = QApplication.instance()
        if not app:
            return
        
        colors = self.get_theme_colors()
        
        # Create QPalette with theme colors
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(colors.background))
        palette.setColor(QPalette.WindowText, QColor(colors.text))
        
        # Base colors (input fields, etc.)
        palette.setColor(QPalette.Base, QColor(colors.background_alternate))
        palette.setColor(QPalette.AlternateBase, QColor(colors.background))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(colors.text))
        palette.setColor(QPalette.BrightText, QColor(colors.text_selected))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(colors.background_alternate))
        palette.setColor(QPalette.ButtonText, QColor(colors.text))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(colors.background_selected))
        palette.setColor(QPalette.HighlightedText, QColor(colors.text_selected))
        
        # Disabled colors
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(colors.text_disabled))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(colors.text_disabled))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(colors.text_disabled))
        
        app.setPalette(palette)
    
    def get_stylesheet(self, theme_name: str = None) -> str:
        """Get complete stylesheet for the theme"""
        colors = self.get_theme_colors(theme_name)
        
        return f"""
        /* Main application style */
        QMainWindow {{
            background-color: {colors.background};
            color: {colors.text};
        }}
        
        /* Tab widget styling */
        QTabWidget::pane {{
            border: 1px solid {colors.border};
            background-color: {colors.background};
        }}
        
        QTabBar::tab {{
            background-color: {colors.background_alternate};
            color: {colors.text};
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid {colors.border};
            border-bottom: none;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors.background_selected};
            color: {colors.text_selected};
        }}
        
        QTabBar::tab:hover {{
            background-color: {colors.accent_light};
        }}
        
        /* Group box styling */
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {colors.border};
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
        }}
        
        /* Button styling */
        QPushButton {{
            background-color: {colors.background_alternate};
            border: 1px solid {colors.border};
            padding: 6px 12px;
            border-radius: 3px;
            color: {colors.text};
        }}
        
        QPushButton:hover {{
            background-color: {colors.accent_light};
        }}
        
        QPushButton:pressed {{
            background-color: {colors.accent_dark};
        }}
        
        QPushButton:disabled {{
            color: {colors.text_disabled};
            background-color: {colors.background};
        }}
        
        /* Input widget styling */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {colors.background_alternate};
            border: 1px solid {colors.border};
            padding: 4px;
            border-radius: 2px;
            color: {colors.text};
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 2px solid {colors.accent};
        }}
        
        /* Combo box dropdown */
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid {colors.text};
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors.background_alternate};
            border: 1px solid {colors.border};
            selection-background-color: {colors.background_selected};
        }}
        
        /* Slider styling */
        QSlider::groove:horizontal {{
            border: 1px solid {colors.border};
            height: 6px;
            background: {colors.background_alternate};
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {colors.accent};
            border: 1px solid {colors.border};
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {colors.accent_light};
        }}
        
        /* Check box styling */
        QCheckBox {{
            color: {colors.text};
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {colors.border};
            border-radius: 2px;
            background: {colors.background_alternate};
        }}
        
        QCheckBox::indicator:checked {{
            background: {colors.accent};
            border: 1px solid {colors.accent_dark};
        }}
        
        /* Progress bar styling */
        QProgressBar {{
            border: 1px solid {colors.border};
            border-radius: 2px;
            text-align: center;
            background: {colors.background_alternate};
        }}
        
        QProgressBar::chunk {{
            background-color: {colors.accent};
            border-radius: 1px;
        }}
        
        /* Scroll bar styling */
        QScrollBar:vertical {{
            background: {colors.background_alternate};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {colors.border};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {colors.accent};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* Status bar styling */
        QStatusBar {{
            background-color: {colors.background_alternate};
            border-top: 1px solid {colors.border};
            color: {colors.text};
        }}
        
        /* Dock widget styling */
        QDockWidget {{
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        
        QDockWidget::title {{
            background-color: {colors.background_alternate};
            padding: 4px;
            border-bottom: 1px solid {colors.border};
        }}
        
        /* Tool tip styling */
        QToolTip {{
            background-color: {colors.background_alternate};
            color: {colors.text};
            border: 1px solid {colors.border};
            padding: 4px;
            border-radius: 2px;
        }}
        
        /* Menu styling */
        QMenuBar {{
            background-color: {colors.background};
            color: {colors.text};
            border-bottom: 1px solid {colors.border};
        }}
        
        QMenuBar::item {{
            padding: 4px 8px;
            background: transparent;
        }}
        
        QMenuBar::item:selected {{
            background-color: {colors.background_selected};
        }}
        
        QMenu {{
            background-color: {colors.background_alternate};
            border: 1px solid {colors.border};
        }}
        
        QMenu::item {{
            padding: 4px 16px;
        }}
        
        QMenu::item:selected {{
            background-color: {colors.background_selected};
        }}
        
        /* Text edit styling */
        QTextEdit {{
            background-color: {colors.background_alternate};
            border: 1px solid {colors.border};
            color: {colors.text};
        }}
        
        /* Splitter styling */
        QSplitter::handle {{
            background-color: {colors.border};
        }}
        
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        """
    
    def get_plot_colors(self, theme_name: str = None) -> Dict[str, str]:
        """Get plot-specific colors for PyQtGraph"""
        colors = self.get_theme_colors(theme_name)
        
        return {
            'background': colors.plot_background,
            'foreground': colors.text,
            'grid': colors.plot_grid,
            'axis': colors.plot_axis,
            'line': colors.plot_line,
            'pen': colors.accent,
            'brush': colors.accent_light
        }

# Global theme manager instance
theme_manager = ThemeManager()