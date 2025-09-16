"""
About Dialog for RF Spectrum Analyzer
Displays application information, version, credits, and system details
"""

import sys
import platform
import subprocess
from typing import Dict, List, Tuple
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QPushButton, QTextEdit, QScrollArea, QGroupBox,
    QFormLayout, QDialogButtonBox, QApplication, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QPainter, QIcon

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('about_dialog')

# Application metadata
APP_NAME = "RF Spectrum Analyzer"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Advanced Software Defined Radio Spectrum Analysis Tool"
APP_COPYRIGHT = "© 2024 RF Spectrum Analyzer Team"
APP_LICENSE = "MIT License"
APP_WEBSITE = "https://github.com/rf-spectrum-analyzer"

CREDITS = [
    ("Development Team", [
        "Lead Developer: AI Assistant",
        "UI/UX Design: PySide6 Framework",
        "DSP Implementation: NumPy, SciPy Teams"
    ]),
    ("Libraries and Dependencies", [
        "PySide6: Qt for Python GUI framework",
        "PyQtGraph: Scientific graphics and GUI library",
        "NumPy: Fundamental package for scientific computing",
        "SciPy: Scientific computing library",
        "pyspectrum: Spectrum analysis library",
        "mhostetter/sdr: Software Defined Radio library",
        "scikit-dsp-comm: Digital signal processing library"
    ]),
    ("Hardware Support", [
        "RTL-SDR: RTL2832U based USB dongles",
        "HackRF: HackRF One software defined radio",
        "PlutoSDR: Analog Devices ADALM-PLUTO",
        "SoapySDR: Vendor neutral SDR support library"
    ]),
    ("Special Thanks", [
        "GNU Radio community",
        "Open source SDR community",
        "Python scientific computing community",
        "Qt/PySide6 developers"
    ])
]

class SystemInfoThread(QThread):
    """Thread to collect system information"""
    
    info_ready = Signal(dict)
    
    def run(self):
        """Collect system information in background"""
        try:
            info = self.collect_system_info()
            self.info_ready.emit(info)
        except Exception as e:
            logger.error(f"Failed to collect system info: {str(e)}")
            self.info_ready.emit({})
    
    def collect_system_info(self) -> Dict[str, str]:
        """Collect comprehensive system information"""
        info = {}
        
        # Python information
        info['Python Version'] = sys.version.split()[0]
        info['Python Implementation'] = platform.python_implementation()
        info['Python Compiler'] = platform.python_compiler()
        
        # System information
        info['Operating System'] = platform.system()
        info['OS Version'] = platform.version()
        info['OS Release'] = platform.release()
        info['Architecture'] = platform.machine()
        info['Processor'] = platform.processor()
        
        # Qt/PySide information
        try:
            from PySide6 import __version__ as pyside_version
            info['PySide6 Version'] = pyside_version
        except:
            info['PySide6 Version'] = 'Unknown'
        
        try:
            from PySide6.QtCore import qVersion
            info['Qt Version'] = qVersion()
        except:
            info['Qt Version'] = 'Unknown'
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['Total Memory'] = f"{memory.total / (1024**3):.1f} GB"
            info['Available Memory'] = f"{memory.available / (1024**3):.1f} GB"
            info['CPU Cores'] = str(psutil.cpu_count())
        except ImportError:
            info['Memory Info'] = 'psutil not available'
        
        # Graphics information
        try:
            import OpenGL.GL as gl
            info['OpenGL Version'] = gl.glGetString(gl.GL_VERSION).decode()
            info['OpenGL Vendor'] = gl.glGetString(gl.GL_VENDOR).decode()
            info['OpenGL Renderer'] = gl.glGetString(gl.GL_RENDERER).decode()
        except:
            info['OpenGL'] = 'Not available'
        
        # Installed packages
        installed_packages = self.get_installed_packages()
        for pkg_name, version in installed_packages.items():
            info[f'Package: {pkg_name}'] = version
        
        return info
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get versions of important installed packages"""
        packages = {
            'numpy': None,
            'scipy': None,
            'pyqtgraph': None,
            'matplotlib': None,
            'h5py': None,
            'rtlsdr': None,
            'hackrf': None,
            'adi': None,
            'SoapySDR': None
        }
        
        for pkg_name in packages:
            try:
                if pkg_name == 'rtlsdr':
                    import rtlsdr
                    packages[pkg_name] = getattr(rtlsdr, '__version__', 'Unknown')
                elif pkg_name == 'hackrf':
                    import hackrf
                    packages[pkg_name] = getattr(hackrf, '__version__', 'Unknown')
                elif pkg_name == 'adi':
                    import adi
                    packages[pkg_name] = getattr(adi, '__version__', 'Unknown')
                elif pkg_name == 'SoapySDR':
                    import SoapySDR
                    packages[pkg_name] = getattr(SoapySDR, '__version__', 'Unknown')
                else:
                    module = __import__(pkg_name)
                    packages[pkg_name] = getattr(module, '__version__', 'Unknown')
            except ImportError:
                packages[pkg_name] = 'Not installed'
            except Exception:
                packages[pkg_name] = 'Error'
        
        return packages

class AboutDialog(QDialog):
    """About dialog with tabbed interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setModal(True)
        self.resize(600, 500)
        
        self.system_info = {}
        self.setup_ui()
        self.load_system_info()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.about_tab = self.create_about_tab()
        self.credits_tab = self.create_credits_tab()
        self.system_tab = self.create_system_tab()
        self.license_tab = self.create_license_tab()
        
        self.tab_widget.addTab(self.about_tab, "About")
        self.tab_widget.addTab(self.credits_tab, "Credits")
        self.tab_widget.addTab(self.system_tab, "System Info")
        self.tab_widget.addTab(self.license_tab, "License")
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
    
    def create_about_tab(self) -> QWidget:
        """Create the about tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Application logo/icon (placeholder)
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("QLabel { background-color: #2E3440; border: 1px solid #4C566A; }")
        icon_label.setFixedSize(128, 128)
        icon_label.setText("📡")  # Placeholder icon
        font = QFont()
        font.setPointSize(48)
        icon_label.setFont(font)
        layout.addWidget(icon_label)
        
        # Application name and version
        name_label = QLabel(f"<h1>{APP_NAME}</h1>")
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        
        version_label = QLabel(f"<h3>Version {APP_VERSION}</h3>")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # Description
        desc_label = QLabel(f"<p>{APP_DESCRIPTION}</p>")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Copyright and website
        copyright_label = QLabel(f"<p>{APP_COPYRIGHT}</p>")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        website_label = QLabel(f'<p><a href="{APP_WEBSITE}">{APP_WEBSITE}</a></p>')
        website_label.setAlignment(Qt.AlignCenter)
        website_label.setOpenExternalLinks(True)
        layout.addWidget(website_label)
        
        # Features list
        features_group = QGroupBox("Key Features")
        features_layout = QVBoxLayout(features_group)
        
        features = [
            "• Real-time spectrum analysis and visualization",
            "• Support for multiple SDR hardware platforms",
            "• Advanced signal processing capabilities",
            "• Waterfall display with customizable colormaps",
            "• Peak detection and measurement tools",
            "• Data export in multiple formats",
            "• Configurable FFT and averaging settings",
            "• Modern, responsive user interface"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            features_layout.addWidget(feature_label)
        
        layout.addWidget(features_group)
        layout.addStretch()
        
        return widget
    
    def create_credits_tab(self) -> QWidget:
        """Create the credits tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        for section_name, items in CREDITS:
            section_group = QGroupBox(section_name)
            section_layout = QVBoxLayout(section_group)
            
            for item in items:
                item_label = QLabel(item)
                item_label.setWordWrap(True)
                section_layout.addWidget(item_label)
            
            scroll_layout.addWidget(section_group)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_system_tab(self) -> QWidget:
        """Create the system information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress bar for loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)
        
        # System info display
        self.system_info_text = QTextEdit()
        self.system_info_text.setReadOnly(True)
        self.system_info_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.system_info_text)
        
        # Copy button
        copy_button = QPushButton("Copy System Info to Clipboard")
        copy_button.clicked.connect(self.copy_system_info)
        layout.addWidget(copy_button)
        
        return widget
    
    def create_license_tab(self) -> QWidget:
        """Create the license tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setPlainText(self.get_license_text())
        layout.addWidget(license_text)
        
        return widget
    
    def get_license_text(self) -> str:
        """Get the license text"""
        return """MIT License

Copyright (c) 2024 RF Spectrum Analyzer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-Party Licenses:

This software uses several third-party libraries, each with their own licenses:

- PySide6: GNU Lesser General Public License (LGPL) v3
- PyQtGraph: MIT License
- NumPy: BSD License
- SciPy: BSD License
- pyspectrum: MIT License (if applicable)
- mhostetter/sdr: MIT License (if applicable)
- scikit-dsp-comm: BSD License (if applicable)

Please refer to the individual library documentation for complete license terms.
"""
    
    def load_system_info(self):
        """Load system information in background"""
        self.system_info_thread = SystemInfoThread()
        self.system_info_thread.info_ready.connect(self.on_system_info_ready)
        self.system_info_thread.start()
    
    def on_system_info_ready(self, info: Dict[str, str]):
        """Handle system info ready signal"""
        self.system_info = info
        self.progress_bar.hide()
        self.display_system_info()
    
    def display_system_info(self):
        """Display system information in text widget"""
        if not self.system_info:
            self.system_info_text.setPlainText("Failed to collect system information.")
            return
        
        text_lines = []
        text_lines.append(f"{APP_NAME} {APP_VERSION} - System Information")
        text_lines.append("=" * 60)
        text_lines.append("")
        
        # Group related information
        groups = {
            "Application": ["App"],
            "Python Environment": ["Python"],
            "Operating System": ["Operating", "OS", "Architecture", "Processor"],
            "GUI Framework": ["PySide6", "Qt"],
            "System Resources": ["Memory", "CPU"],
            "Graphics": ["OpenGL"],
            "Installed Packages": ["Package:"]
        }
        
        for group_name, keywords in groups.items():
            group_items = []
            for key, value in self.system_info.items():
                if any(keyword in key for keyword in keywords):
                    group_items.append(f"  {key}: {value}")
            
            if group_items:
                text_lines.append(f"{group_name}:")
                text_lines.extend(group_items)
                text_lines.append("")
        
        # Add any remaining items
        remaining_items = []
        for key, value in self.system_info.items():
            found_in_group = False
            for keywords in groups.values():
                if any(keyword in key for keyword in keywords):
                    found_in_group = True
                    break
            if not found_in_group:
                remaining_items.append(f"  {key}: {value}")
        
        if remaining_items:
            text_lines.append("Other:")
            text_lines.extend(remaining_items)
        
        self.system_info_text.setPlainText("\n".join(text_lines))
    
    def copy_system_info(self):
        """Copy system information to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.system_info_text.toPlainText())
        
        # Show temporary message (could be improved with a status bar)
        logger.info("System information copied to clipboard")

def show_about_dialog(parent=None):
    """Convenience function to show the about dialog"""
    dialog = AboutDialog(parent)
    dialog.exec()