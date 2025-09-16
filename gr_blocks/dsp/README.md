# RF Spectrum Analyzer

Advanced RF signal acquisition, analysis, and processing software that integrates three powerful libraries:

- **[pyspectrum](https://github.com/naj1024/pyspectrum)**: Real-time FFT spectrum analysis and SDR input
- **[mhostetter/sdr](https://github.com/mhostetter/sdr)**: Digital signal processing, filtering, modulation/demodulation  
- **[scikit-dsp-comm](https://github.com/mwickert/scikit-dsp-comm)**: Advanced DSP algorithms, synchronization, FEC coding

Built with **PySide6** for the GUI and **PyQtGraph** for high-performance real-time plotting.

## Features

### SDR Hardware Support
- **RTL-SDR**: Popular USB SDR dongles (24 MHz - 1.8 GHz)
- **PlutoSDR**: Analog Devices ADALM-Pluto (70 MHz - 6 GHz)
- **HackRF One**: Wide-band SDR with TX capability (1 MHz - 6 GHz) 
- **SoapySDR**: Universal SDR interface supporting many devices
- **Audio**: Sound card input for low-frequency analysis
- **File Input**: Support for various IQ file formats

### Real-Time Signal Processing
- **FFT Spectrum Analysis**: Real-time spectrum display with configurable FFT size, windowing, and averaging
- **Waterfall Display**: Time-frequency visualization
- **Digital Filtering**: FIR, IIR, polyphase, and adaptive filters
- **Modulation Analysis**: PSK, QAM, FSK, MSK, AM, FM detection and demodulation
- **I/Q Constellation**: Real-time constellation diagrams
- **Signal Analysis**: Power, bandwidth, SNR, THD measurements

### Advanced DSP Capabilities
- **Filter Design**: Kaiser, Butterworth, Chebyshev, Elliptical filters
- **Resampling**: Rational resampling, interpolation, decimation
- **Synchronization**: Carrier recovery, symbol timing recovery, PLLs
- **Error Correction**: Convolutional coding with Viterbi decoding
- **Sequence Generation**: Gold, Barker, Zadoff-Chu sequences

### Professional GUI
- **Multi-tab Interface**: Spectrum, I/Q analysis, constellation, logs
- **Real-time Controls**: Frequency, gain, sample rate, filtering
- **Performance Monitoring**: CPU usage, sample rates, buffer status
- **Device Management**: Auto-discovery and configuration
- **Export Capabilities**: Screenshots, data export, reports

## Installation

### Prerequisites
- Python 3.9 or newer
- Qt6 development libraries
- SDR hardware drivers (optional)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/rfspectrum/rf-spectrum-analyzer.git
cd rf-spectrum-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development with all features
pip install -e .[dev,docs]
```

### SDR Driver Installation

#### RTL-SDR
```bash
# Ubuntu/Debian
sudo apt-get install rtl-sdr librtlsdr-dev

# Windows: Download from https://github.com/rtlsdrblog/rtl-sdr-blog
pip install pyrtlsdr
```

#### PlutoSDR
```bash
pip install pyadi-iio
# Follow Analog Devices setup guide for drivers
```

#### HackRF
```bash
# Ubuntu/Debian
sudo apt-get install hackrf libhackrf-dev

# Install Python bindings (various options available)
pip install pyhackrf  # or other HackRF Python library
```

#### SoapySDR (Universal)
```bash
# Ubuntu/Debian
sudo apt-get install soapysdr-tools libsoapysdr-dev

pip install pysoapysdr
```

## Quick Start

### Command Line Usage

```bash
# Basic usage with auto-detected device
rf-spectrum-analyzer

# Specify device and frequency
rf-spectrum-analyzer --source rtlsdr --freq 433.92e6 --samplerate 2e6

# Use PlutoSDR
rf-spectrum-analyzer --source pluto --freq 915e6 --gain 30

# Process IQ file
rf-spectrum-analyzer --source file --input signal.wav

# Headless mode for batch processing
rf-spectrum-analyzer --headless --freq 1e9 --log-level DEBUG
```

### GUI Usage

1. **Launch the application:**
   ```bash
   rf-spectrum-analyzer
   ```

2. **Select your SDR device** from the dropdown menu

3. **Configure parameters:**
   - Center frequency (MHz)
   - Sample rate (MSps) 
   - RF gain (dB)
   - Processing settings

4. **Click Start** to begin real-time analysis

5. **Use tabs to view:**
   - Spectrum: FFT and waterfall plots
   - I/Q Analysis: Constellation and time domain
   - Logs: Application messages

### Python API

```python
from rf_spectrum_analyzer.core.app import RFSpectrumApp
from rf_spectrum_analyzer.config.settings import AppSettings

# Create settings
settings = AppSettings()
settings.sdr.center_freq = 433.92e6
settings.sdr.sample_rate = 2e6

# Initialize application  
app = RFSpectrumApp(settings)

# Start processing
app.start_processing()
```

## Configuration

The application uses YAML configuration files. The main config is located at:
`rf_spectrum_analyzer/config/default_config.yaml`

### Key Configuration Sections

```yaml
sdr:
  source: "auto"              # Device type
  center_freq: 433920000.0    # Hz
  sample_rate: 2000000.0      # Hz
  gain: 30.0                  # dB

dsp:
  fft_size: 2048             # Must be power of 2
  window: "hanning"          # Window function
  averaging: 10              # Number of averages

gui:
  theme: "auto"              # auto, light, dark
  update_rate: 30            # FPS

processing:
  enable_real_time: true
  buffer_size: 4096
  num_buffers: 16
```

## Architecture

```
rf_spectrum_analyzer/
├── main.py                    # Application entry point
├── config/
│   ├── settings.py           # Configuration management
│   └── default_config.yaml   # Default settings
├── core/
│   ├── app.py               # Main application class
│   ├── sdr_backend.py       # SDR backend management
│   └── signal_processor.py  # Signal processing engine
├── backends/
│   ├── soapy_backend.py     # SoapySDR support
│   ├── rtlsdr_backend.py    # RTL-SDR support
│   ├── pluto_backend.py     # PlutoSDR support
│   └── hackrf_backend.py    # HackRF support
├── gui/
│   ├── main_window.py       # Main GUI window
│   └── dialogs/             # Dialog windows
├── dsp/
│   ├── filters.py           # Digital filter implementations
│   ├── modulation.py        # Modulation/demodulation
│   ├── analysis.py          # Signal analysis tools
│   └── utils.py             # DSP utilities
└── utils/
    ├── logger.py            # Logging utilities
    ├── helpers.py           # Helper functions
    └── file_io.py           # File I/O operations
```

## Key Classes

- **`RFSpectrumApp`**: Main application coordinator
- **`SDRBackendManager`**: Manages multiple SDR devices
- **`SignalProcessor`**: Core DSP processing engine
- **`FilterBank`**: Digital filter collection (FIR/IIR)
- **`ModulationAnalyzer`**: Modulation detection/demodulation
- **`MainWindow`**: GUI interface with real-time plots

## Performance Features

- **Multi-threading**: Separate threads for acquisition, processing, and GUI
- **Optimized FFT**: Uses FFTW library when available
- **Circular Buffers**: Efficient sample management
- **GPU Acceleration**: CUDA support for intensive processing
- **Memory Management**: Configurable buffer sizes and limits

## Supported File Formats

### Input
- **WAV**: Audio files (real samples)
- **IQ**: Raw I/Q binary files 
- **SigMF**: Signal Metadata Format
- **HackRF**: HackRF specific format
- **USRP**: GNU Radio formats

### Output  
- **SigMF**: Standard format with metadata
- **CSV**: Spectrum/analysis data
- **NumPy**: .npy/.npz arrays
- **MATLAB**: .mat files
- **Screenshots**: PNG/PDF plots

## Development

### Project Structure
The codebase follows a modular architecture with clear separation between:
- Hardware abstraction (backends)
- Signal processing (dsp)
- User interface (gui)  
- Configuration (config)
- Utilities (utils)

### Adding New SDR Support
1. Create new backend in `backends/` inheriting from `SDRBackend`
2. Implement required methods: `open()`, `close()`, `read_samples()`, etc.
3. Register backend in `SDRBackendManager`
4. Add device enumeration support

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code follows style guidelines
5. Submit pull request

### Testing
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=rf_spectrum_analyzer tests/

# Test specific backend
pytest tests/test_rtlsdr_backend.py
```

## Troubleshooting

### Common Issues

**"No SDR devices found"**
- Check device drivers are installed
- Verify device permissions (may need udev rules on Linux)
- Try different USB ports
- Check `lsusb` output on Linux

**"ImportError: No module named 'sdr'"**
```bash
pip install sdr
```

**"Qt platform plugin could not be loaded"**
```bash
# On Linux
sudo apt-get install qt6-base-dev

# Set Qt plugin path if needed
export QT_QPA_PLATFORM_PLUGIN_PATH=/path/to/qt/plugins
```

**Poor performance/dropping samples**
- Reduce FFT size
- Lower update rate
- Increase buffer sizes
- Close other applications

### Debug Mode
```bash
rf-spectrum-analyzer --debug --verbose --log-level DEBUG
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[pyspectrum](https://github.com/naj1024/pyspectrum)**: Real-time spectrum analysis framework
- **[mhostetter/sdr](https://github.com/mhostetter/sdr)**: Comprehensive SDR signal processing library  
- **[scikit-dsp-comm](https://github.com/mwickert/scikit-dsp-comm)**: Advanced DSP and communications algorithms
- **[PyQtGraph](https://pyqtgraph.readthedocs.io/)**: High-performance plotting library
- **[PySide6](https://doc.qt.io/qtforpython/)**: Python Qt bindings

## Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive API documentation
- **Examples**: Sample code and tutorials
- **Community**: Discussion forum for users and developers

## Roadmap

### Version 1.1
- [ ] Advanced modulation classification using ML
- [ ] Plugin system for custom processing
- [ ] Remote operation via web interface
- [ ] USRP support via UHD
- [ ] Advanced synchronization algorithms

### Version 1.2  
- [ ] Multi-channel support
- [ ] Direction finding capabilities
- [ ] Protocol analysis (Bluetooth, WiFi, etc.)
- [ ] Real-time demodulation to audio
- [ ] Automated measurement routines

---

**RF Spectrum Analyzer** - Professional SDR signal analysis for research, education, and development.
