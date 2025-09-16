# RF Spectrum Analyzer

A comprehensive Software Defined Radio (SDR) spectrum analysis application built with Python, PySide6, and PyQtGraph. This application provides real-time spectrum analysis and visualization capabilities for various SDR hardware platforms.

## Features

### 🔊 Real-Time Signal Analysis
- **Live Spectrum Display**: Real-time FFT-based spectrum analysis with customizable parameters
- **Waterfall Display**: Time-frequency visualization with multiple colormap options
- **Peak Detection**: Automatic peak identification and measurement
- **Signal Measurements**: Frequency, power, bandwidth, and SNR measurements

### 📡 Multi-Platform SDR Support
- **RTL-SDR**: RTL2832U-based USB dongles
- **HackRF**: HackRF One software defined radio
- **PlutoSDR**: Analog Devices ADALM-PLUTO
- **SoapySDR**: Universal SDR interface supporting multiple vendors

### 🎛️ Advanced Processing
- **Configurable FFT**: Variable FFT sizes, window functions, and overlap settings
- **Digital Filtering**: Multiple filter types and configurations
- **Signal Averaging**: Linear, exponential, and peak-hold averaging modes
- **Real-Time Processing**: Optimized multi-threaded signal processing pipeline

### 💾 Data Management
- **Multiple Export Formats**: CSV, JSON, MATLAB, HDF5, and plain text
- **Configuration Management**: Save and restore application settings
- **Session Recording**: Capture and replay spectrum data
- **Metadata Support**: Comprehensive recording of measurement parameters

### 🎨 Modern Interface
- **Responsive Design**: Clean, modern user interface with PySide6
- **Theme Support**: Dark and light themes with customizable colors
- **Interactive Plots**: Zoom, pan, and cursor measurements with PyQtGraph
- **Dockable Panels**: Flexible workspace organization

## Installation

### Prerequisites
- Python 3.8 or newer
- Qt6 development libraries (usually installed with PySide6)

### Basic Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/rf-spectrum-analyzer.git
   cd rf-spectrum-analyzer
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install SDR libraries (optional, based on your hardware):**

   **For RTL-SDR:**
   ```bash
   # Install RTL-SDR drivers first
   pip install pyrtlsdr
   ```

   **For HackRF:**
   ```bash
   # Install HackRF drivers first
   pip install pyhackrf
   ```

   **For PlutoSDR:**
   ```bash
   # Install libiio first
   pip install pyadi-iio
   ```

   **For SoapySDR:**
   ```bash
   # Install SoapySDR system package first
   # Then install Python bindings (system-dependent)
   ```

### Hardware Setup

#### RTL-SDR
1. Install RTL-SDR drivers using Zadig (Windows) or package manager (Linux/macOS)
2. Ensure device is recognized and accessible
3. May require running with administrator privileges on Windows

#### HackRF
1. Install HackRF drivers and firmware
2. Update firmware to latest version if needed
3. Ensure USB permissions are properly configured (Linux)

#### PlutoSDR
1. Connect via USB or Ethernet
2. Configure network settings if using Ethernet connection
3. Install Analog Devices IIO library

#### SoapySDR
1. Install SoapySDR base library
2. Install appropriate hardware modules (SoapyRTLSDR, SoapyHackRF, etc.)
3. Test hardware detection with `SoapySDRUtil --find`

## Usage

### Starting the Application

```bash
python main.py
```

Or with specific options:
```bash
python main.py --device rtlsdr --center-freq 100MHz --sample-rate 2.4MHz
```

### Command Line Options

- `--device`: SDR device type (rtlsdr, hackrf, pluto, soapy)
- `--device-id`: Specific device identifier
- `--center-freq`: Initial center frequency (Hz, kHz, MHz, GHz)
- `--sample-rate`: Sample rate (Hz, kHz, MHz)
- `--gain`: Initial gain setting (dB)
- `--config`: Load configuration file
- `--theme`: UI theme (dark, light, auto)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Basic Operation

1. **Device Selection**: Choose your SDR device from the controls panel
2. **Frequency Setting**: Set center frequency and sample rate
3. **Gain Control**: Adjust gain manually or enable automatic gain control
4. **Start Analysis**: Click the play button to begin real-time analysis
5. **Adjust Display**: Modify reference level, dynamic range, and averaging settings
6. **Export Data**: Save spectrum data in various formats

### Advanced Features

#### Custom Filters
- Configure digital filters in the processing settings
- Choose from multiple filter types (lowpass, highpass, bandpass, bandstop)
- Adjust filter parameters in real-time

#### Peak Detection
- Enable automatic peak detection
- Configure detection threshold and minimum separation
- View peak frequency and power measurements

#### Data Export
- Export current spectrum or waterfall data
- Multiple format support with metadata
- Batch export capabilities

## Configuration

### Settings File
Application settings are automatically saved to:
- Windows: `%USERPROFILE%\.rf_spectrum_analyzer\config.json`
- Linux/macOS: `~/.rf_spectrum_analyzer/config.json`

### Customization
- Modify themes in `resources/themes.py`
- Add custom colormaps for waterfall display
- Configure keyboard shortcuts
- Customize plot styling

## Development

### Project Structure
```
rf_spectrum_analyzer/
├── main.py                 # Application entry point
├── core/                   # Core application modules
│   ├── app.py             # Main application class
│   ├── sdr_backend.py     # SDR abstraction layer
│   └── signal_processor.py # Signal processing pipeline
├── backends/              # SDR hardware backends
│   ├── rtlsdr_backend.py  # RTL-SDR implementation
│   ├── hackrf_backend.py  # HackRF implementation
│   ├── pluto_backend.py   # PlutoSDR implementation
│   └── soapy_backend.py   # SoapySDR implementation
├── gui/                   # User interface components
│   ├── main_window.py     # Main application window
│   ├── spectrum_widget.py # Spectrum display
│   ├── waterfall_widget.py # Waterfall display
│   └── controls_widget.py # Control panels
├── dsp/                   # Digital signal processing
│   ├── filters.py         # Filter implementations
│   ├── modulation.py      # Modulation/demodulation
│   ├── analysis.py        # Signal analysis tools
│   └── utils.py           # DSP utilities
├── dialogs/               # Dialog windows
│   ├── settings_dialog.py # Application settings
│   └── about_dialog.py    # About dialog
├── utils/                 # Utility modules
│   ├── file_io.py         # Data import/export
│   ├── logger.py          # Logging system
│   └── helpers.py         # Helper functions
├── resources/             # UI resources
│   ├── themes.py          # Theme definitions
│   └── icons.py           # Icon generation
└── config/                # Configuration
    └── settings.py        # Settings dataclasses
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Testing

Run tests with pytest:
```bash
pytest tests/
```

For GUI tests:
```bash
pytest tests/gui/ --qt-api=pyside6
```

## Troubleshooting

### Common Issues

#### Device Not Detected
- Verify hardware drivers are installed
- Check USB connections and permissions
- Try running with administrator/root privileges
- Use `SoapySDRUtil --find` to test SoapySDR devices

#### Poor Performance
- Reduce FFT size or update rate
- Enable GPU acceleration if available
- Close other applications using SDR hardware
- Check for USB bandwidth limitations

#### Import Errors
- Ensure all required packages are installed
- Check for conflicting package versions
- Try creating a virtual environment
- Verify Qt6 installation

### Log Files
Application logs are saved to:
- Windows: `%USERPROFILE%\.rf_spectrum_analyzer\logs\`
- Linux/macOS: `~/.rf_spectrum_analyzer/logs/`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PySide6/Qt**: Modern cross-platform GUI framework
- **PyQtGraph**: High-performance scientific graphics
- **NumPy/SciPy**: Scientific computing foundation
- **GNU Radio**: Inspiration for SDR signal processing
- **SDR Hardware Communities**: RTL-SDR, HackRF, PlutoSDR, and SoapySDR projects

## Support

- **Documentation**: [Project Wiki](https://github.com/your-username/rf-spectrum-analyzer/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/rf-spectrum-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/rf-spectrum-analyzer/discussions)

## Roadmap

### Planned Features
- [ ] Additional demodulation modes (AM, FM, SSB)
- [ ] Network remote control API
- [ ] Plugin system for custom processing
- [ ] Advanced signal analysis tools
- [ ] Multi-channel support
- [ ] Recording and playback capabilities
- [ ] Frequency scanning modes
- [ ] Signal classification algorithms

### Performance Improvements
- [ ] GPU-accelerated FFT processing
- [ ] Optimized memory management
- [ ] Multi-threading enhancements
- [ ] Real-time streaming protocols

---

**RF Spectrum Analyzer** - Professional SDR spectrum analysis made accessible.