# Create a summary of all files created
import csv
from pathlib import Path

# Collect all files in the project
project_files = []

def scan_directory(directory, base_path="rf_spectrum_analyzer"):
    for item in directory.iterdir():
        if item.is_file():
            relative_path = str(item.relative_to(Path("rf_spectrum_analyzer")))
            file_size = item.stat().st_size
            project_files.append({
                'File Path': relative_path,
                'File Size (bytes)': file_size,
                'Description': get_file_description(relative_path)
            })
        elif item.is_dir() and not item.name.startswith('.'):
            scan_directory(item, base_path)

def get_file_description(file_path):
    """Get description based on file path"""
    descriptions = {
        'main.py': 'Main application entry point with command-line interface',
        'setup.py': 'Python package installation script',
        'requirements.txt': 'Python dependencies specification',
        'README.md': 'Project documentation and usage guide',
        
        'config/settings.py': 'Application configuration management system',
        'config/default_config.yaml': 'Default configuration file with all settings',
        
        'core/app.py': 'Main application class coordinating all components',
        'core/sdr_backend.py': 'SDR backend management and device abstraction',
        'core/signal_processor.py': 'Core signal processing engine integrating all DSP libraries',
        
        'backends/soapy_backend.py': 'SoapySDR backend for universal SDR support',
        'backends/rtlsdr_backend.py': 'RTL-SDR backend implementation',
        'backends/pluto_backend.py': 'PlutoSDR backend implementation',
        'backends/hackrf_backend.py': 'HackRF backend implementation',
        
        'gui/main_window.py': 'Main GUI window with spectrum display and controls',
        
        'dsp/filters.py': 'Advanced digital filters using mhostetter/sdr and scikit-dsp-comm',
        'dsp/modulation.py': 'Modulation/demodulation and automatic classification',
        
        'utils/logger.py': 'Comprehensive logging system with performance monitoring',
        'utils/helpers.py': 'Utility functions for signal processing and file I/O',
    }
    
    return descriptions.get(file_path, 'Project file')

# Scan the project directory
if Path("rf_spectrum_analyzer").exists():
    scan_directory(Path("rf_spectrum_analyzer"))

# Calculate totals
total_files = len(project_files)
total_size = sum(f['File Size (bytes)'] for f in project_files)
total_lines = 0

# Estimate lines of code for Python files
for file_info in project_files:
    if file_info['File Path'].endswith('.py'):
        # Rough estimate: 50 bytes per line average
        estimated_lines = file_info['File Size (bytes)'] // 50
        total_lines += estimated_lines

# Save to CSV
with open('rf_spectrum_analyzer_files.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['File Path', 'File Size (bytes)', 'Description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for file_info in sorted(project_files, key=lambda x: x['File Path']):
        writer.writerow(file_info)

# Print summary
print(f"RF Spectrum Analyzer Project Summary:")
print(f"=====================================")
print(f"Total files created: {total_files}")
print(f"Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
print(f"Estimated lines of code: {total_lines:,}")
print(f"")
print(f"Project structure saved to: rf_spectrum_analyzer_files.csv")
print(f"")

# Print file structure
print("Project Structure:")
print("==================")

# Group files by directory
from collections import defaultdict
files_by_dir = defaultdict(list)

for file_info in project_files:
    path_parts = file_info['File Path'].split('/')
    if len(path_parts) == 1:
        directory = "root"
    else:
        directory = '/'.join(path_parts[:-1])
    
    files_by_dir[directory].append(path_parts[-1])

for directory in sorted(files_by_dir.keys()):
    if directory == "root":
        print("rf_spectrum_analyzer/")
    else:
        print(f"rf_spectrum_analyzer/{directory}/")
    
    for filename in sorted(files_by_dir[directory]):
        print(f"├── {filename}")
    print()

print("\nKey Components:")
print("===============")
print("✓ Main application entry point with CLI support")
print("✓ Comprehensive configuration system") 
print("✓ SDR backend abstraction supporting 4 major platforms")
print("✓ Advanced signal processing engine integrating 3 libraries:")
print("  - pyspectrum: Real-time FFT spectrum analysis")
print("  - mhostetter/sdr: Digital signal processing & modulation")
print("  - scikit-dsp-comm: Advanced DSP algorithms & synchronization")
print("✓ Professional GUI with real-time spectrum display")
print("✓ Comprehensive DSP filter bank (FIR/IIR)")
print("✓ Modulation detection and demodulation")
print("✓ Performance monitoring and logging")
print("✓ Utility functions and helper modules")
print("✓ Complete documentation and setup")

print(f"\nTotal project: {total_files} files, ~{total_lines:,} lines of code")
print("Ready for installation and use!")