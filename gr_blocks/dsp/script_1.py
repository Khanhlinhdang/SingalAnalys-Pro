# Create requirements.txt with all necessary dependencies
requirements_content = """# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0

# GUI and plotting
PySide6>=6.3.0
pyqtgraph>=0.13.0

# SDR and signal processing libraries
sdr>=0.5.0
scikit-dsp-comm>=2.0.0

# FFT libraries
pyfftw>=0.12.0

# SDR backends
SoapySDR>=0.8.1
pyrtlsdr>=0.2.9

# Pluto SDR support
pyadi-iio>=0.0.14
iio>=0.24

# Audio support
sounddevice>=0.4.0
PyAudio>=0.2.11

# Network and protocols
websockets>=10.0
paho-mqtt>=1.6.0

# File and data handling
h5py>=3.7.0
sigmf>=1.0.0

# HID support for FunCube
hidapi>=0.12.0

# USB monitoring
psutil>=5.9.0

# Logging and configuration
PyYAML>=6.0
"""

with open("rf_spectrum_analyzer/requirements.txt", "w") as f:
    f.write(requirements_content)

print("Created requirements.txt")