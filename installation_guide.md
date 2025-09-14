# Installation and Usage Guide for Inmarsat AERO Receiver
# Python implementation for USRP N210 + OQPSK demodulation

## System Requirements

### Hardware Requirements
- USRP N210 with appropriate daughterboard (WBX, SBX, or UBX)
- L-band antenna (1540-1550 MHz) with RHCP polarization
- Computer with Gigabit Ethernet port
- Sufficient processing power (multi-core CPU recommended)

### Software Requirements
- Python 3.8 or higher
- Operating System: Linux (Ubuntu/Debian recommended), Windows, or macOS

## Installation Steps

### 1. Install UHD (USRP Hardware Driver)

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install libuhd-dev uhd-host
sudo uhd_images_downloader
```

#### Windows:
Download UHD installer from Ettus Research website:
https://www.ettus.com/sdr-software/detail/usrp-hardware-driver

#### macOS:
```bash
brew install uhd
```

### 2. Install Python Dependencies

```bash
pip install numpy scipy pyqtgraph PySide6 uhd
```

### 3. Verify USRP Connection

```bash
uhd_find_devices
uhd_usrp_probe
```

## File Structure

```
inmarsat_aero_receiver/
├── inmarsat_aero_receiver.py    # Core receiver and demodulator classes
├── inmarsat_gui.py              # GUI interface with PyQtGraph
├── installation_guide.md        # This file
├── usage_examples.py            # Example usage scripts
└── README.md                    # Project overview
```

## Usage Instructions

### 1. Command Line Usage

#### Basic Reception Test:
```python
from inmarsat_aero_receiver import InmarsatAEROReceiver

# Create receiver
receiver = InmarsatAEROReceiver(
    center_freq=1545e6,      # 1545 MHz
    usrp_sample_rate=2.4e6,  # 2.4 MS/s
    demod_sample_rate=48000  # 48 kS/s
)

# Start reception for 60 seconds
results = receiver.start_reception(duration_seconds=60)

# Save I/Q data
receiver.save_iq_data("inmarsat_data.csv")

# Print results
print(f"Symbols processed: {results['symbols_processed']}")
print(f"Signal quality: {results['demod_quality']}")
```

### 2. GUI Application

#### Launch GUI:
```bash
python inmarsat_gui.py
```

#### GUI Features:
- Real-time spectrum analyzer (1540-1550 MHz)
- Waterfall display
- OQPSK constellation diagram
- USRP control (frequency, gain, sample rate)
- Demodulator settings
- I/Q data export
- Live signal quality monitoring

### 3. USRP N210 Configuration

#### Recommended Settings:
- **Center Frequency**: 1545 MHz (Inmarsat L-band)
- **Sample Rate**: 2.4 MS/s (allows 10 MHz bandwidth)
- **Gain**: 20-40 dB (adjust based on signal strength)
- **Antenna**: RHCP L-band antenna
- **Daughterboard**: WBX (50 MHz - 2.2 GHz) recommended

#### Example USRP Setup:
```python
import uhd

# Create USRP device
usrp = uhd.usrp.MultiUSRP()

# Configure for Inmarsat AERO
usrp.set_rx_rate(2.4e6, 0)              # 2.4 MS/s
usrp.set_rx_freq(1545e6, 0)             # 1545 MHz
usrp.set_rx_gain(30, 0)                 # 30 dB gain
usrp.set_rx_antenna("RX2", 0)           # Use RX2 port

# Capture samples
samples = usrp.recv_num_samps(
    num_samps=240000,     # 100ms at 2.4 MS/s
    freq=1545e6,
    rate=2.4e6,
    channels=[0],
    gain=30
)
```

## Signal Processing Pipeline

### 1. RF Reception (USRP N210)
- Frequency: 1540-1550 MHz (L-band)
- Bandwidth: Up to 40 MHz (depending on daughterboard)
- Sample Rate: 2.4 MS/s (configurable)
- Data Format: Complex float32 (I/Q samples)

### 2. Signal Detection
- FFT-based spectrum analysis
- Peak detection above noise floor
- Signal bandwidth estimation
- Frequency offset measurement

### 3. OQPSK Demodulation
- **AGC**: Automatic gain control with 4-second averaging
- **RRC Filtering**: Root raised cosine with α=1.0
- **Symbol Timing**: Early-late gate with IIR resonator
- **Carrier Tracking**: BPSK 2x method for phase/frequency
- **OQPSK Decode**: Q-channel delay compensation
- **Output**: Soft bits (0-255 range) + signal quality metrics

### 4. Data Output
- I/Q soft bits (CSV format)
- Signal quality metrics (MSE)
- Constellation diagrams
- Spectrum waterfall data

## Configuration Parameters

### USRP Settings
```python
usrp_config = {
    'center_freq': 1545e6,        # Hz
    'sample_rate': 2.4e6,         # Samples/second
    'gain': 30,                   # dB
    'antenna': 'RX2',             # Antenna port
    'bandwidth': 40e6             # Hz (optional)
}
```

### Demodulator Settings
```python
demod_config = {
    'sample_rate': 48000,         # Demodulator sample rate
    'symbol_rate': 10500,         # OQPSK symbol rate (bps)
    'signal_threshold': 0.5,      # MSE threshold for signal detection
    'rrc_alpha': 1.0,            # RRC filter roll-off
    'rrc_taps': 55               # RRC filter length
}
```

## Troubleshooting

### 1. USRP Connection Issues
```bash
# Check USB/Ethernet connection
uhd_find_devices

# Test USRP functionality
uhd_usrp_probe --args="addr=192.168.10.2"

# Check firmware/FPGA images
uhd_images_downloader
```

### 2. No Signal Detection
- Verify antenna connection and polarization (RHCP)
- Check frequency range (1540-1550 MHz)
- Adjust gain settings (try 20-50 dB range)
- Verify satellite visibility and coverage
- Check for local interference

### 3. Poor Demodulation Quality
- Increase signal threshold if too sensitive
- Adjust AGC parameters for signal level
- Check symbol timing recovery performance
- Verify carrier frequency accuracy
- Monitor constellation diagram for proper alignment

### 4. Performance Issues
- Reduce sample rate if CPU overloaded
- Increase buffer sizes for better throughput
- Use faster computer or optimize code
- Monitor for buffer overruns in UHD

## Signal Analysis

### Inmarsat AERO Signal Characteristics
- **Frequency Band**: L-band (1525-1660 MHz)
- **Downlink**: 1545-1547 MHz (satellite to aircraft)
- **Modulation**: OQPSK at 10.5 kbps (main channel)
- **Also supports**: 600 bps and 1200 bps BPSK
- **Polarization**: Right-Hand Circular Polarized (RHCP)
- **Channel Spacing**: Variable, burst-based

### Expected Signal Levels
- **Strong Signal**: > -70 dBm (MSE < 0.3)
- **Good Signal**: -70 to -80 dBm (MSE < 0.5)  
- **Weak Signal**: -80 to -90 dBm (MSE 0.5-1.0)
- **Threshold**: < -90 dBm (MSE > 1.0, unreliable)

## Data Export Formats

### CSV I/Q Data Format
```csv
Symbol_Index,I_Soft,Q_Soft,MSE,Timestamp
1,145,132,0.234567,1694123456.789012
2,156,128,0.198765,1694123456.789045
...
```

### Signal Quality Metrics
- **MSE**: Mean Square Error (constellation deviation)
- **AGC Gain**: Automatic gain control level
- **Carrier Phase**: Phase tracking error
- **Symbol Timing**: Timing recovery metrics

## Advanced Usage

### Custom Signal Processing
```python
# Access raw demodulator for custom processing
demod = OQPSKDemodulator(sample_rate=48000, symbol_rate=10500)

for complex_sample in raw_samples:
    result = demod.process_sample(complex_sample)
    
    if result['symbol_ready']:
        # Custom processing of demodulated symbols
        i_symbol = result['i_soft']
        q_symbol = result['q_soft']
        quality = result['mse']
        
        # Your custom processing here...
```

### Batch Processing
```python
# Process recorded files
import numpy as np

# Load complex samples from file
samples = np.load('recorded_samples.npy')

# Process through receiver
receiver = InmarsatAEROReceiver()
for chunk in np.array_split(samples, 100):
    # Process chunk...
```

## Performance Optimization

### Real-time Processing
- Use adequate buffer sizes (50-100ms)
- Monitor CPU usage and adjust sample rates
- Consider multi-threading for parallel processing
- Optimize decimation factors

### Memory Management
- Limit constellation buffer size
- Periodically clear old I/Q data
- Use circular buffers for spectrum data
- Monitor memory usage in long runs

## Support and Resources

### Documentation
- UHD Manual: https://files.ettus.com/manual/
- GNU Radio Tutorials: https://wiki.gnuradio.org/
- JAERO Project: https://github.com/jontio/JAERO

### Community
- USRP Users Group
- Software Defined Radio communities
- Amateur radio digital mode groups

### Hardware Suppliers
- Ettus Research (USRP devices)
- RF antenna manufacturers for L-band
- SDR component suppliers