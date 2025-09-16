#!/usr/bin/env python3
"""
Test file for Bitstream Dock functionality in RF Spectrum Analyzer
Tests the docker operations (float, hide, show, menu integration) for bitstream display.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest

from rf_spectrum_analyzer.gui.main_window import MainWindow
from rf_spectrum_analyzer.config.settings import Settings


class BitstreamDockTester(QWidget):
    """Standalone tester for bitstream dock functionality."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitstream Dock Tester")
        self.setGeometry(100, 100, 300, 200)
        
        # Load default settings
        self.settings = Settings()
        
        # Create main window with bitstream dock
        self.main_window = MainWindow(self.settings)
        self.main_window.show()
        
        # Setup test UI
        self.setup_ui()
        
        # Data generation timer
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.generate_test_data)
        self.bit_counter = 0
        
    def setup_ui(self):
        """Setup test control UI."""
        layout = QVBoxLayout(self)
        
        # Test controls
        self.start_data_btn = QPushButton("Start Test Data")
        self.start_data_btn.clicked.connect(self.start_test_data)
        layout.addWidget(self.start_data_btn)
        
        self.stop_data_btn = QPushButton("Stop Test Data")
        self.stop_data_btn.clicked.connect(self.stop_test_data)
        layout.addWidget(self.stop_data_btn)
        
        # Dock controls
        self.toggle_bitstream_btn = QPushButton("Toggle Bitstream Dock")
        self.toggle_bitstream_btn.clicked.connect(self.toggle_bitstream_dock)
        layout.addWidget(self.toggle_bitstream_btn)
        
        self.float_bitstream_btn = QPushButton("Float/Dock Bitstream")
        self.float_bitstream_btn.clicked.connect(self.float_bitstream_dock)
        layout.addWidget(self.float_bitstream_btn)
        
        self.reset_layout_btn = QPushButton("Reset Layout")
        self.reset_layout_btn.clicked.connect(self.reset_layout)
        layout.addWidget(self.reset_layout_btn)
        
        # Test patterns
        self.pattern_btn = QPushButton("Send Test Pattern")
        self.pattern_btn.clicked.connect(self.send_test_pattern)
        layout.addWidget(self.pattern_btn)
        
        self.random_btn = QPushButton("Send Random Data")
        self.random_btn.clicked.connect(self.send_random_data)
        layout.addWidget(self.random_btn)
        
    def start_test_data(self):
        """Start generating test bitstream data."""
        print("✅ Starting test data generation...")
        self.data_timer.start(100)  # Generate data every 100ms
        self.start_data_btn.setEnabled(False)
        self.stop_data_btn.setEnabled(True)
        
    def stop_test_data(self):
        """Stop generating test data."""
        print("🛑 Stopping test data generation...")
        self.data_timer.stop()
        self.start_data_btn.setEnabled(True)
        self.stop_data_btn.setEnabled(False)
        
    def generate_test_data(self):
        """Generate continuous test bitstream data."""
        # Generate different patterns
        pattern_type = (self.bit_counter // 50) % 4
        
        if pattern_type == 0:
            # Alternating pattern
            bits = [self.bit_counter % 2] * 10
        elif pattern_type == 1:
            # Block pattern
            bits = [1] * 5 + [0] * 5
        elif pattern_type == 2:
            # Random pattern
            bits = np.random.randint(0, 2, 8).tolist()
        else:
            # Sequence pattern
            bits = [(i % 8) // 4 for i in range(self.bit_counter, self.bit_counter + 6)]
        
        # Send to bitstream display
        self.main_window.update_bitstream(np.array(bits, dtype=np.uint8))
        self.bit_counter += len(bits)
        
    def toggle_bitstream_dock(self):
        """Test bitstream dock visibility toggle."""
        print("🔄 Testing bitstream dock toggle...")
        self.main_window.toggle_bitstream_dock()
        
        # Check state
        visible = self.main_window.bitstream_dock.isVisible()
        print(f"   Bitstream dock visible: {visible}")
        
    def float_bitstream_dock(self):
        """Test bitstream dock floating toggle."""
        print("🚀 Testing bitstream dock float/dock...")
        
        current_floating = self.main_window.bitstream_dock.isFloating()
        self.main_window.bitstream_dock.setFloating(not current_floating)
        
        new_floating = self.main_window.bitstream_dock.isFloating()
        print(f"   Bitstream dock floating: {current_floating} → {new_floating}")
        
    def reset_layout(self):
        """Test layout reset functionality."""
        print("🔄 Testing layout reset...")
        self.main_window.reset_dock_layout()
        
        # Verify both docks are visible and docked
        const_visible = self.main_window.constellation_dock.isVisible()
        const_floating = self.main_window.constellation_dock.isFloating()
        bit_visible = self.main_window.bitstream_dock.isVisible()
        bit_floating = self.main_window.bitstream_dock.isFloating()
        
        print(f"   Constellation: visible={const_visible}, floating={const_floating}")
        print(f"   Bitstream: visible={bit_visible}, floating={bit_floating}")
        
    def send_test_pattern(self):
        """Send a specific test pattern."""
        print("📊 Sending test pattern...")
        
        # Create test pattern: header + data + footer
        header = [1, 0, 1, 0, 1, 1, 0, 0]  # 0xAC
        data = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]  # 0xD371
        footer = [0, 1, 0, 1, 0, 1, 0, 1]  # 0x55
        
        pattern = header + data + footer
        self.main_window.update_bitstream(np.array(pattern, dtype=np.uint8))
        print(f"   Sent {len(pattern)} bits: {pattern}")
        
    def send_random_data(self):
        """Send random bitstream data."""
        print("🎲 Sending random data...")
        
        # Generate random bits
        num_bits = np.random.randint(20, 100)
        random_bits = np.random.randint(0, 2, num_bits)
        
        self.main_window.update_bitstream(random_bits)
        print(f"   Sent {num_bits} random bits")
        
        # Display first 20 bits for reference
        display_bits = random_bits[:20].tolist()
        print(f"   First 20 bits: {display_bits}")


def test_dock_properties():
    """Test basic dock widget properties."""
    print("\n🧪 Testing dock widget properties...")
    
    settings = Settings()
    main_window = MainWindow(settings)
    
    # Test bitstream dock exists
    assert hasattr(main_window, 'bitstream_dock'), "❌ Bitstream dock not found"
    assert hasattr(main_window, 'bitstream_widget'), "❌ Bitstream widget not found"
    print("✅ Bitstream dock and widget created")
    
    # Test dock properties
    dock = main_window.bitstream_dock
    features = dock.features()
    
    assert features & dock.DockWidgetMovable, "❌ Dock not movable"
    assert features & dock.DockWidgetFloatable, "❌ Dock not floatable"
    assert features & dock.DockWidgetClosable, "❌ Dock not closable"
    print("✅ Dock features configured correctly")
    
    # Test initial state
    assert dock.isVisible(), "❌ Dock not initially visible"
    assert not dock.isFloating(), "❌ Dock initially floating"
    print("✅ Initial dock state correct")
    
    # Test widget functionality
    widget = main_window.bitstream_widget
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    widget.add_bits(test_bits)
    assert len(widget.bit_buffer) == 8, "❌ Bits not added correctly"
    print("✅ Widget accepts bit data")
    
    main_window.close()
    print("✅ All dock property tests passed!")


def test_menu_integration():
    """Test menu integration for bitstream dock."""
    print("\n🧪 Testing menu integration...")
    
    settings = Settings()
    main_window = MainWindow(settings)
    main_window.show()
    
    # Test menu exists
    assert hasattr(main_window, 'view_menu'), "❌ View menu not found"
    assert hasattr(main_window, 'bitstream_action'), "❌ Bitstream action not found"
    print("✅ Menu and actions created")
    
    # Test action properties
    action = main_window.bitstream_action
    assert action.isCheckable(), "❌ Action not checkable"
    assert action.isChecked(), "❌ Action not initially checked"
    assert action.shortcut().toString() == "Ctrl+B", "❌ Shortcut not correct"
    print("✅ Action properties correct")
    
    # Test action functionality
    initial_visible = main_window.bitstream_dock.isVisible()
    action.trigger()  # Should toggle visibility
    new_visible = main_window.bitstream_dock.isVisible()
    assert initial_visible != new_visible, "❌ Action doesn't toggle visibility"
    print("✅ Action toggles dock visibility")
    
    main_window.close()
    print("✅ All menu integration tests passed!")


def run_automated_tests():
    """Run automated tests for bitstream dock."""
    print("🚀 Starting automated bitstream dock tests...\n")
    
    try:
        test_dock_properties()
        test_menu_integration()
        print("\n🎉 All automated tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test application."""
    app = QApplication(sys.argv)
    
    print("🔬 Bitstream Dock Functionality Tester")
    print("=" * 50)
    
    # Run automated tests first
    if not run_automated_tests():
        return 1
    
    print("\n" + "=" * 50)
    print("🎮 Starting interactive test interface...")
    print("Use the buttons to test dock operations:")
    print("  • Toggle visibility (Ctrl+B)")
    print("  • Float/dock the window")
    print("  • Reset layout (Ctrl+R)")
    print("  • Generate test bitstream data")
    
    # Create interactive tester
    tester = BitstreamDockTester()
    tester.show()
    
    print("\n✅ Interactive tester started")
    print("   Main Window: RF Spectrum Analyzer with bitstream dock")
    print("   Control Window: Test controls")
    print("   Close control window to exit")
    
    # Run application
    exit_code = app.exec()
    
    print("\n👋 Test session completed")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())