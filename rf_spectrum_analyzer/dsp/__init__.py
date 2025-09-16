"""
Digital Signal Processing Module
Advanced DSP functions for RF signal analysis
"""

from rf_spectrum_analyzer.dsp.filters import *
from rf_spectrum_analyzer.dsp.modulation import *
from rf_spectrum_analyzer.dsp.analysis import *
from rf_spectrum_analyzer.dsp.utils import *
from rf_spectrum_analyzer.dsp.modulation_analysis import *
from rf_spectrum_analyzer.dsp.demodulation_engine import *
from rf_spectrum_analyzer.dsp.decoding_engine import *

__all__ = [
    # Filters
    'FIRFilter', 'IIRFilter', 'PolyphaseFilter', 'AdaptiveFilter',
    'ButterworthFilter', 'ChebyshevFilter', 'EllipticFilter',
    'design_lowpass', 'design_bandpass', 'design_highpass', 'design_bandstop',
    
    # Modulation
    'PSKModulator', 'QAMModulator', 'FSKModulator', 'OFDMModulator',
    'PSKDemodulator', 'QAMDemodulator', 'FSKDemodulator', 'OFDMDemodulator',
    'CPMModulator', 'MSKModulator', 'AMModulator', 'FMModulator',
    
    # Analysis
    'SpectrumAnalyzer', 'SignalDetector', 'ParameterEstimator',
    'StatisticalAnalyzer', 'InterferenceAnalyzer', 'PhasePlaneAnalyzer',
    'ConstellationAnalyzer', 'EyeDiagramAnalyzer', 'PowerMeasurement',
    
    # Utilities
    'WindowFunction', 'NoiseGenerator', 'ChannelModel',
    'FrequencyDomainTools', 'TimeDomainTools', 'StatisticalTools',
    'PerformanceMetrics', 'SignalQualityMetrics',
    
    # New Modulation Analysis and Demodulation
    'ModulationAnalyzer', 'EncodingAnalyzer', 'DemodulationEngine', 'DecodingEngine',
    'create_modulation_analyzer', 'create_encoding_analyzer', 
    'create_demodulation_engine', 'create_decoding_engine'
]