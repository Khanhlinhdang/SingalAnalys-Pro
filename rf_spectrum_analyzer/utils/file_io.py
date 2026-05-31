"""
File I/O utilities for RF Spectrum Analyzer
Handles data export, import, and file format conversions
"""

import os
import json
import csv
import pickle
import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('file_io')

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False

class DataExporter:
    """Export spectrum data to various file formats"""
    
    SUPPORTED_FORMATS = ['csv', 'json', 'mat', 'h5', 'pkl', 'txt']
    
    def __init__(self):
        self.metadata = {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata to be included in exported files"""
        self.metadata = metadata
        
    def export_spectrum_data(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filename: str,
        format: str = 'csv',
        **kwargs
    ) -> bool:
        """
        Export spectrum data to file
        
        Args:
            frequencies: Frequency array in Hz
            power_spectrum: Power spectrum in dB
            filename: Output filename
            format: Export format ('csv', 'json', 'mat', 'h5', 'pkl', 'txt')
            **kwargs: Additional format-specific options
            
        Returns:
            Success status
        """
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                return self._export_csv(frequencies, power_spectrum, filepath, **kwargs)
            elif format.lower() == 'json':
                return self._export_json(frequencies, power_spectrum, filepath, **kwargs)
            elif format.lower() == 'mat':
                return self._export_mat(frequencies, power_spectrum, filepath, **kwargs)
            elif format.lower() == 'h5':
                return self._export_h5(frequencies, power_spectrum, filepath, **kwargs)
            elif format.lower() == 'pkl':
                return self._export_pickle(frequencies, power_spectrum, filepath, **kwargs)
            elif format.lower() == 'txt':
                return self._export_txt(frequencies, power_spectrum, filepath, **kwargs)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def _export_csv(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path,
        delimiter: str = ',',
        include_header: bool = True
    ) -> bool:
        """Export to CSV format"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            if include_header:
                # Write metadata as comments
                for key, value in self.metadata.items():
                    writer.writerow([f'# {key}: {value}'])
                
                # Write column headers
                writer.writerow(['Frequency_Hz', 'Power_dB'])
            
            # Write data
            for freq, power in zip(frequencies, power_spectrum):
                writer.writerow([freq, power])
        
        logger.info(f"Exported spectrum data to CSV: {filepath}")
        return True
    
    def _export_json(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path,
        indent: int = 2
    ) -> bool:
        """Export to JSON format"""
        data = {
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat(),
            'frequencies': frequencies.tolist(),
            'power_spectrum': power_spectrum.tolist(),
            'units': {
                'frequency': 'Hz',
                'power': 'dB'
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        
        logger.info(f"Exported spectrum data to JSON: {filepath}")
        return True
    
    def _export_mat(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path
    ) -> bool:
        """Export to MATLAB .mat format"""
        data = {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        scipy.io.savemat(filepath, data)
        logger.info(f"Exported spectrum data to MAT: {filepath}")
        return True
    
    def _export_h5(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path,
        compression: str = 'gzip'
    ) -> bool:
        """Export to HDF5 format"""
        if not H5PY_AVAILABLE:
            logger.error("h5py is not installed. Cannot export HDF5 format.")
            return False

        with h5py.File(filepath, 'w') as f:
            # Create datasets
            f.create_dataset('frequencies', data=frequencies, compression=compression)
            f.create_dataset('power_spectrum', data=power_spectrum, compression=compression)
            
            # Add metadata as attributes
            for key, value in self.metadata.items():
                f.attrs[key] = value
            
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['units_frequency'] = 'Hz'
            f.attrs['units_power'] = 'dB'
        
        logger.info(f"Exported spectrum data to HDF5: {filepath}")
        return True
    
    def _export_pickle(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path
    ) -> bool:
        """Export to Python pickle format"""
        data = {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Exported spectrum data to pickle: {filepath}")
        return True
    
    def _export_txt(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        filepath: Path,
        delimiter: str = '\t'
    ) -> bool:
        """Export to plain text format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write metadata as comments
            for key, value in self.metadata.items():
                f.write(f"# {key}: {value}\n")
            
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Frequency_Hz{delimiter}Power_dB\n")
            
            # Write data
            for freq, power in zip(frequencies, power_spectrum):
                f.write(f"{freq}{delimiter}{power}\n")
        
        logger.info(f"Exported spectrum data to TXT: {filepath}")
        return True

class DataImporter:
    """Import spectrum data from various file formats"""
    
    def __init__(self):
        self.last_metadata = {}
    
    def import_spectrum_data(
        self,
        filename: str,
        format: str = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Import spectrum data from file
        
        Args:
            filename: Input filename
            format: File format (auto-detected if None)
            
        Returns:
            Tuple of (frequencies, power_spectrum, metadata)
        """
        try:
            filepath = Path(filename)
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return None, None, {}
            
            # Auto-detect format from extension
            if format is None:
                format = filepath.suffix.lower().lstrip('.')
            
            if format == 'csv':
                return self._import_csv(filepath)
            elif format == 'json':
                return self._import_json(filepath)
            elif format == 'mat':
                return self._import_mat(filepath)
            elif format == 'h5':
                return self._import_h5(filepath)
            elif format == 'pkl':
                return self._import_pickle(filepath)
            elif format == 'txt':
                return self._import_txt(filepath)
            else:
                logger.error(f"Unsupported import format: {format}")
                return None, None, {}
                
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            return None, None, {}
    
    def _import_csv(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from CSV format"""
        metadata = {}
        frequencies = []
        power_spectrum = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in reader:
                if not row:
                    continue
                
                # Parse metadata comments
                if row[0].startswith('#'):
                    if ':' in row[0]:
                        key, value = row[0][1:].split(':', 1)
                        metadata[key.strip()] = value.strip()
                    continue
                
                # Skip header row
                if row[0].lower().startswith('frequency'):
                    continue
                
                # Parse data
                try:
                    freq = float(row[0])
                    power = float(row[1])
                    frequencies.append(freq)
                    power_spectrum.append(power)
                except (ValueError, IndexError):
                    continue
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from CSV: {filepath}")
        
        return np.array(frequencies), np.array(power_spectrum), metadata
    
    def _import_json(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from JSON format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        frequencies = np.array(data['frequencies'])
        power_spectrum = np.array(data['power_spectrum'])
        metadata = data.get('metadata', {})
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from JSON: {filepath}")
        
        return frequencies, power_spectrum, metadata
    
    def _import_mat(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from MATLAB .mat format"""
        data = scipy.io.loadmat(filepath)
        
        frequencies = data['frequencies'].flatten()
        power_spectrum = data['power_spectrum'].flatten()
        metadata = {}
        
        # Extract metadata if available
        if 'metadata' in data:
            metadata = data['metadata']
            if hasattr(metadata, 'item'):
                metadata = metadata.item()
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from MAT: {filepath}")
        
        return frequencies, power_spectrum, metadata
    
    def _import_h5(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from HDF5 format"""
        if not H5PY_AVAILABLE:
            logger.error("h5py is not installed. Cannot import HDF5 format.")
            return np.array([]), np.array([]), {}

        with h5py.File(filepath, 'r') as f:
            frequencies = f['frequencies'][:]
            power_spectrum = f['power_spectrum'][:]
            
            # Extract metadata from attributes
            metadata = {}
            for key, value in f.attrs.items():
                metadata[key] = value
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from HDF5: {filepath}")
        
        return frequencies, power_spectrum, metadata
    
    def _import_pickle(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from Python pickle format"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        frequencies = data['frequencies']
        power_spectrum = data['power_spectrum']
        metadata = data.get('metadata', {})
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from pickle: {filepath}")
        
        return frequencies, power_spectrum, metadata
    
    def _import_txt(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Import from plain text format"""
        metadata = {}
        frequencies = []
        power_spectrum = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse metadata comments
                if line.startswith('#'):
                    if ':' in line:
                        key, value = line[1:].split(':', 1)
                        metadata[key.strip()] = value.strip()
                    continue
                
                # Parse data
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        freq = float(parts[0])
                        power = float(parts[1])
                        frequencies.append(freq)
                        power_spectrum.append(power)
                except ValueError:
                    continue
        
        self.last_metadata = metadata
        logger.info(f"Imported spectrum data from TXT: {filepath}")
        
        return np.array(frequencies), np.array(power_spectrum), metadata

class ConfigurationManager:
    """Manage application configuration files"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            self.config_dir = Path.home() / '.rf_spectrum_analyzer'
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / 'config.json'
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                logger.info("No configuration file found, using defaults")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return {}
    
    def backup_config(self) -> bool:
        """Create a backup of the current configuration"""
        try:
            if self.config_file.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = self.config_dir / f'config_backup_{timestamp}.json'
                
                import shutil
                shutil.copy2(self.config_file, backup_file)
                
                logger.info(f"Configuration backed up to {backup_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {str(e)}")
            return False

# Utility functions
def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes"""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0.0

def ensure_directory(directory: Union[str, Path]):
    """Ensure directory exists, create if necessary"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_available_formats() -> List[str]:
    """Get list of available export/import formats"""
    return DataExporter.SUPPORTED_FORMATS.copy()

def validate_filename(filename: str) -> str:
    """Validate and sanitize filename"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename