"""
File I/O utilities for RF Spectrum Analyzer
Handles data export, import, and file format conversions
"""

import os
import json
import csv
import pickle
import wave
import numpy as np
import scipy.io
from scipy.io import wavfile
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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

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

    def export_artifact_image(
        self,
        artifact: Dict[str, Any],
        filename: str,
        format: Optional[str] = None,
    ) -> bool:
        """Export decoded image artifact (e.g. NOAA APT) in png/json/npy formats."""
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            ext = (format or filepath.suffix.lstrip('.')).lower()
            if not ext:
                ext = 'png'

            payload = artifact.get('payload', artifact)
            matrix = payload.get('image_matrix') or payload.get('preview_rows')
            if matrix is None:
                logger.error("Artifact does not contain image_matrix/preview_rows")
                return False

            image = np.asarray(matrix, dtype=np.uint8)
            if image.size == 0:
                logger.error("Artifact image matrix is empty")
                return False

            if ext in {'png', 'jpg', 'jpeg'}:
                if not MATPLOTLIB_AVAILABLE:
                    logger.error("matplotlib is not installed. Cannot export image format.")
                    return False
                plt.imsave(filepath, image, cmap='gray', vmin=0, vmax=255)
                logger.info(f"Exported artifact image to {filepath}")
                return True

            if ext == 'json':
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'metadata': self.metadata,
                    'artifact': {
                        'type': artifact.get('type', 'image'),
                        'confidence': artifact.get('confidence', 0.0),
                        'payload': payload,
                    },
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Exported artifact image JSON to {filepath}")
                return True

            if ext in {'npy', 'npz'}:
                if ext == 'npz':
                    np.savez_compressed(filepath, image=image, metadata=self.metadata)
                else:
                    np.save(filepath, image)
                logger.info(f"Exported artifact image array to {filepath}")
                return True

            logger.error(f"Unsupported artifact image format: {ext}")
            return False

        except Exception as e:
            logger.error(f"Artifact image export failed: {str(e)}")
            return False

    def export_decode_session_report(
        self,
        records: List[Dict[str, Any]],
        filename: str,
        format: str = 'json',
    ) -> bool:
        """Export session-level decode report with trend series and artifact references."""
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            export_format = (format or filepath.suffix.lstrip('.')).lower() or 'json'
            if export_format != 'json':
                logger.error(f"Unsupported decode session report format: {export_format}")
                return False

            timestamps = [r.get('timestamp') for r in records]
            snr_series = [self._to_float(r.get('snr')) for r in records]
            decode_quality = [r.get('decode_quality', {}) if isinstance(r, dict) else {} for r in records]

            report = {
                'timestamp': datetime.now().isoformat(),
                'metadata': self.metadata,
                'record_count': len(records),
                'session_summary': {
                    'avg_snr': self._series_average(snr_series),
                    'avg_ber': self._series_average([self._to_float(q.get('ber')) for q in decode_quality]),
                    'avg_per': self._series_average([self._to_float(q.get('per')) for q in decode_quality]),
                    'avg_crc_ok_rate': self._series_average([self._to_float(q.get('crc_ok_rate')) for q in decode_quality]),
                    'avg_frame_lock_ratio': self._series_average([self._to_float(q.get('frame_lock_ratio')) for q in decode_quality]),
                    'total_artifacts': int(sum((q.get('artifact_count') or 0) for q in decode_quality)),
                },
                'trends': {
                    'timestamps': timestamps,
                    'snr': snr_series,
                    'ber': [self._to_float(q.get('ber')) for q in decode_quality],
                    'per': [self._to_float(q.get('per')) for q in decode_quality],
                    'crc_ok_rate': [self._to_float(q.get('crc_ok_rate')) for q in decode_quality],
                    'frame_lock_ratio': [self._to_float(q.get('frame_lock_ratio')) for q in decode_quality],
                    'artifact_count': [int(q.get('artifact_count') or 0) for q in decode_quality],
                    'frame_count': [int(q.get('frame_count') or 0) for q in decode_quality],
                },
                'records': records,
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Exported decode session report to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Decode session report export failed: {str(e)}")
            return False

    def export_pcm_wav_from_artifact(self, artifact: Dict[str, Any], filename: str) -> bool:
        """Export PCM artifact payload to WAV file."""
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            payload = artifact.get('payload', artifact)
            if not isinstance(payload, dict):
                logger.error("Invalid PCM artifact payload")
                return False

            samples = payload.get('samples')
            sample_rate = int(payload.get('sample_rate') or 0)
            channels = int(payload.get('channels') or 1)

            if samples is None or sample_rate <= 0:
                logger.error("PCM artifact missing samples/sample_rate")
                return False

            pcm = np.asarray(samples, dtype=np.int16)
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(max(1, channels))
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())

            logger.info(f"Exported PCM WAV to {filepath}")
            return True
        except Exception as e:
            logger.error(f"PCM WAV export failed: {str(e)}")
            return False

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _series_average(self, values: List[Optional[float]]) -> Optional[float]:
        usable = [v for v in values if v is not None]
        if not usable:
            return None
        return float(np.mean(usable))

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

    def import_signal_source(
        self,
        filename: str,
        format: str = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Import a raw signal source file into complex IQ samples."""
        try:
            filepath = Path(filename)
            if not filepath.exists():
                logger.error(f"Signal source file not found: {filepath}")
                return None, {}

            source_format = (format or filepath.suffix.lower().lstrip('.')).lower()
            if source_format in {'wav', 'wave'}:
                return self._import_wav_signal(filepath)
            if source_format == 'npy':
                return self._import_npy_signal(filepath)
            if source_format == 'npz':
                return self._import_npz_signal(filepath)

            logger.error(f"Unsupported signal source format: {source_format}")
            return None, {}
        except Exception as e:
            logger.error(f"Signal source import failed: {str(e)}")
            return None, {}

    def _import_wav_signal(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Import IQ samples from a WAV file."""
        sample_rate, data = wavfile.read(filepath)
        samples = np.asarray(data)
        metadata: Dict[str, Any] = {
            'source_file': str(filepath),
            'source_format': 'wav',
            'sample_rate': float(sample_rate),
            'shape': tuple(samples.shape),
            'dtype': str(samples.dtype),
        }

        if samples.ndim == 1:
            iq_data = samples.astype(np.float32)
            if np.issubdtype(samples.dtype, np.integer):
                scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)) or 1.0
                iq_data = iq_data / scale
            iq_data = iq_data.astype(np.complex64)
            metadata['channels'] = 1
        else:
            channel_count = int(samples.shape[1])
            metadata['channels'] = channel_count
            i_channel = samples[:, 0]
            q_channel = samples[:, 1] if channel_count > 1 else np.zeros_like(i_channel)

            if np.issubdtype(samples.dtype, np.integer):
                scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)) or 1.0
                i_channel = i_channel.astype(np.float32) / scale
                q_channel = q_channel.astype(np.float32) / scale
            else:
                i_channel = i_channel.astype(np.float32)
                q_channel = q_channel.astype(np.float32)

            iq_data = i_channel.astype(np.complex64) + 1j * q_channel.astype(np.complex64)

        metadata['duration_seconds'] = float(len(samples) / sample_rate) if sample_rate else None
        self.last_metadata = metadata
        logger.info(f"Imported signal source from WAV: {filepath}")
        return iq_data.astype(np.complex64), metadata

    def _import_npy_signal(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Import IQ samples from a NumPy .npy file."""
        data = np.load(filepath, allow_pickle=True)
        metadata: Dict[str, Any] = {
            'source_file': str(filepath),
            'source_format': 'npy',
        }

        iq_data = self._coerce_signal_array(data, metadata)
        self.last_metadata = metadata
        logger.info(f"Imported signal source from NPY: {filepath}")
        return iq_data, metadata

    def _import_npz_signal(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Import IQ samples from a NumPy .npz file."""
        archive = np.load(filepath, allow_pickle=True)
        metadata: Dict[str, Any] = {
            'source_file': str(filepath),
            'source_format': 'npz',
            'keys': list(archive.keys()),
        }

        data = None
        for key in ('iq', 'iq_data', 'samples', 'data'):
            if key in archive:
                data = archive[key]
                metadata['data_key'] = key
                break
        if data is None and archive.files:
            metadata['data_key'] = archive.files[0]
            data = archive[archive.files[0]]

        iq_data = self._coerce_signal_array(data, metadata)
        self.last_metadata = metadata
        logger.info(f"Imported signal source from NPZ: {filepath}")
        return iq_data, metadata

    def _coerce_signal_array(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Convert imported arrays into a complex64 IQ stream."""
        if data is None:
            return np.array([], dtype=np.complex64)

        arr = np.asarray(data)
        metadata['shape'] = tuple(arr.shape)
        metadata['dtype'] = str(arr.dtype)

        if arr.ndim == 0:
            arr = arr.reshape(1)

        if np.iscomplexobj(arr):
            return arr.astype(np.complex64).flatten()

        if arr.ndim == 1:
            real = arr.astype(np.float32)
            if np.issubdtype(arr.dtype, np.integer):
                scale = float(max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)) or 1.0
                real = real / scale
            return real.astype(np.complex64)

        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            i_channel = arr[..., 0].astype(np.float32)
            q_channel = arr[..., 1].astype(np.float32)
            if np.issubdtype(arr.dtype, np.integer):
                scale = float(max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)) or 1.0
                i_channel = i_channel / scale
                q_channel = q_channel / scale
            return (i_channel.astype(np.complex64) + 1j * q_channel.astype(np.complex64)).flatten()

        return arr.astype(np.complex64).flatten()

    def import_decode_session_report(self, filename: str) -> Dict[str, Any]:
        """Import a previously exported decode session report JSON."""
        try:
            filepath = Path(filename)
            if not filepath.exists():
                logger.error(f"Decode session report not found: {filepath}")
                return {}

            with open(filepath, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            if not isinstance(payload, dict) or 'records' not in payload:
                logger.error("Invalid decode session report format")
                return {}

            logger.info(f"Imported decode session report from {filepath}")
            return payload
        except Exception as e:
            logger.error(f"Decode session report import failed: {str(e)}")
            return {}
    
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