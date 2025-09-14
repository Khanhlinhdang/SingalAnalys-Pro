# Now create an integration module that connects the burst detector to the existing SDR project

integration_code = '''
"""
Burst Detector Integration Module

Tích hợp burst detector vào hệ thống SDR hiện tại
"""

import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import the burst detector system
try:
    from burst_detector_demodulator import CompleteBurstDetector
    BURST_DETECTOR_AVAILABLE = True
except ImportError:
    BURST_DETECTOR_AVAILABLE = False
    print("Warning: Burst detector module not available")

class BurstDetectionPipeline:
    """Integration pipeline for burst detection trong hệ thống SDR"""
    
    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.processing_active = False
        self.detection_thread = None
        
        # Initialize burst detector if available
        if BURST_DETECTOR_AVAILABLE:
            self.burst_detector = CompleteBurstDetector(
                sample_rate=sample_rate,
                fft_size=1024,
                threshold_db=18,
                history_size=100,
                lookahead=5
            )
            print("✅ Burst Detection Pipeline initialized")
        else:
            self.burst_detector = None
            print("❌ Burst detector not available")
        
        # Results storage
        self.detected_bursts = Queue(maxsize=1000)
        self.processing_stats = {
            'total_samples': 0,
            'bursts_detected': 0,
            'frames_decoded': 0,
            'processing_time': 0.0
        }
        
        # Callbacks for real-time processing
        self.burst_callback = None
        self.frame_callback = None
    
    def set_callbacks(self, burst_callback=None, frame_callback=None):
        """Set callbacks for real-time burst and frame notifications"""
        self.burst_callback = burst_callback
        self.frame_callback = frame_callback
    
    def start_processing(self):
        """Start burst detection processing"""
        if not BURST_DETECTOR_AVAILABLE:
            return False
        
        if self.processing_active:
            return True
        
        self.processing_active = True
        print("🎯 Burst detection processing started")
        return True
    
    def stop_processing(self):
        """Stop burst detection processing"""
        self.processing_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        print("⏹️ Burst detection processing stopped")
    
    def process_signal_chunk(self, iq_samples):
        """Process một chunk IQ samples cho burst detection"""
        if not BURST_DETECTOR_AVAILABLE or not self.processing_active:
            return []
        
        start_time = time.time()
        
        try:
            # Process through burst detector
            detected_frames = self.burst_detector.process_iq_stream(iq_samples)
            
            # Update statistics
            self.processing_stats['total_samples'] += len(iq_samples)
            self.processing_stats['bursts_detected'] += len(detected_frames)
            self.processing_stats['processing_time'] += time.time() - start_time
            
            # Store results
            for frame in detected_frames:
                if not self.detected_bursts.full():
                    self.detected_bursts.put(frame)
                
                # Call frame callback if set
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
            
            # Update frame count
            self.processing_stats['frames_decoded'] += len(detected_frames)
            
            return detected_frames
        
        except Exception as e:
            print(f"Burst processing error: {e}")
            return []
    
    def get_recent_bursts(self, max_count=10):
        """Get recent detected bursts"""
        bursts = []
        count = 0
        
        while not self.detected_bursts.empty() and count < max_count:
            try:
                burst = self.detected_bursts.get_nowait()
                bursts.append(burst)
                count += 1
            except Empty:
                break
        
        return bursts
    
    def get_processing_statistics(self):
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        # Add burst detector stats if available
        if self.burst_detector:
            detector_stats = self.burst_detector.get_statistics()
            stats.update(detector_stats)
        
        # Calculate derived statistics
        if stats['total_samples'] > 0:
            stats['samples_per_second'] = stats['total_samples'] / max(stats['processing_time'], 0.001)
        
        if stats['bursts_detected'] > 0:
            stats['decode_success_rate'] = stats['frames_decoded'] / stats['bursts_detected']
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_samples': 0,
            'bursts_detected': 0,
            'frames_decoded': 0,
            'processing_time': 0.0
        }
        
        if self.burst_detector:
            self.burst_detector.reset_statistics()

class BurstDetectionWidget:
    """UI Widget cho burst detection results"""
    
    def __init__(self):
        self.pipeline = None
        self.update_interval = 1.0  # seconds
        self.last_update = time.time()
        
        # Display state
        self.displayed_frames = []
        self.max_displayed = 50
        
    def set_pipeline(self, pipeline):
        """Set the burst detection pipeline"""
        self.pipeline = pipeline
        
        # Set up callbacks
        if self.pipeline:
            self.pipeline.set_callbacks(
                frame_callback=self.on_frame_detected
            )
    
    def on_frame_detected(self, frame_info):
        """Handle new frame detection"""
        # Add to display list
        self.displayed_frames.append({
            'timestamp': time.time(),
            'frame_type': frame_info.get('frame_type', 'UNKNOWN'),
            'confidence': frame_info.get('confidence', 0.0),
            'bit_count': frame_info.get('bit_count', 0),
            'signal_level': frame_info.get('signal_level', 0.0),
            'burst_id': frame_info.get('burst_id', 'N/A')
        })
        
        # Limit display list size
        if len(self.displayed_frames) > self.max_displayed:
            self.displayed_frames = self.displayed_frames[-self.max_displayed:]
    
    def get_display_data(self):
        """Get data for display in UI"""
        if not self.pipeline:
            return {
                'frames': [],
                'statistics': {},
                'status': 'Not connected'
            }
        
        # Get recent frames
        frames = self.displayed_frames.copy()
        
        # Get statistics
        stats = self.pipeline.get_processing_statistics()
        
        # Determine status
        if self.pipeline.processing_active:
            status = "Active"
        else:
            status = "Stopped"
        
        return {
            'frames': frames,
            'statistics': stats,
            'status': status
        }
    
    def clear_display(self):
        """Clear displayed frames"""
        self.displayed_frames = []

class USRPBurstIntegration:
    """Integration class for USRP burst detection"""
    
    def __init__(self, usrp_interface):
        self.usrp = usrp_interface
        self.pipeline = BurstDetectionPipeline()
        
        # Processing control
        self.processing_thread = None
        self.processing_active = False
        
        print("🔗 USRP Burst Integration ready")
    
    def start_burst_detection(self):
        """Start burst detection with USRP"""
        if not self.usrp or not self.pipeline:
            return False
        
        if self.processing_active:
            return True
        
        self.processing_active = True
        self.pipeline.start_processing()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("🚀 USRP burst detection started")
        return True
    
    def stop_burst_detection(self):
        """Stop burst detection"""
        self.processing_active = False
        self.pipeline.stop_processing()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        print("⏹️ USRP burst detection stopped")
    
    def _processing_loop(self):
        """Main processing loop for USRP samples"""
        chunk_size = 10000  # Process in 10k sample chunks
        
        while self.processing_active:
            try:
                # Get samples from USRP
                if hasattr(self.usrp, 'get_samples'):
                    samples = self.usrp.get_samples(timeout=0.1)
                    
                    if samples is not None and len(samples) >= chunk_size:
                        # Process in chunks
                        for i in range(0, len(samples), chunk_size):
                            chunk = samples[i:i+chunk_size]
                            if len(chunk) >= chunk_size:
                                self.pipeline.process_signal_chunk(chunk)
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                print(f"USRP processing error: {e}")
                time.sleep(0.1)
    
    def get_pipeline(self):
        """Get the burst detection pipeline"""
        return self.pipeline

# Integration with existing enhanced processing pipeline
class EnhancedPipelineWithBurst:
    """Enhanced processing pipeline với burst detection"""
    
    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        
        # Original pipeline components would be imported here
        # from enhanced_processing_pipeline import EnhancedProcessingPipeline
        # self.original_pipeline = EnhancedProcessingPipeline(sample_rate)
        
        # Burst detection pipeline
        self.burst_pipeline = BurstDetectionPipeline(sample_rate)
        
        # Processing modes
        self.processing_mode = 'auto'  # 'auto', 'burst_only', 'traditional_only'
        
        print("🔧 Enhanced Pipeline with Burst Detection ready")
    
    def set_processing_mode(self, mode):
        """Set processing mode: auto, burst_only, traditional_only"""
        if mode in ['auto', 'burst_only', 'traditional_only']:
            self.processing_mode = mode
            print(f"Processing mode set to: {mode}")
    
    def process_signal(self, signal, signal_info=None):
        """Process signal with both traditional and burst detection"""
        results = {
            'traditional_results': None,
            'burst_results': None,
            'combined_analysis': {}
        }
        
        try:
            if self.processing_mode in ['auto', 'burst_only']:
                # Burst detection processing
                burst_frames = self.burst_pipeline.process_signal_chunk(signal)
                results['burst_results'] = {
                    'detected_frames': burst_frames,
                    'statistics': self.burst_pipeline.get_processing_statistics()
                }
            
            if self.processing_mode in ['auto', 'traditional_only']:
                # Traditional processing would go here
                # results['traditional_results'] = self.original_pipeline.process_signal(signal, signal_info)
                results['traditional_results'] = {
                    'status': 'Traditional processing placeholder',
                    'note': 'Would integrate with existing pipeline'
                }
            
            # Combined analysis
            if self.processing_mode == 'auto':
                results['combined_analysis'] = self._combine_results(
                    results['traditional_results'], 
                    results['burst_results']
                )
        
        except Exception as e:
            print(f"Enhanced pipeline error: {e}")
        
        return results
    
    def _combine_results(self, traditional, burst):
        """Combine traditional and burst detection results"""
        combined = {
            'confidence_boost': False,
            'cross_validation': 'pending',
            'recommendation': 'use_traditional'
        }
        
        # If burst detection found frames
        if burst and burst.get('detected_frames'):
            combined['confidence_boost'] = True
            combined['recommendation'] = 'use_burst'
            
            # Analyze frame types
            frame_types = [f.get('frame_type', 'UNKNOWN') for f in burst['detected_frames']]
            combined['detected_frame_types'] = list(set(frame_types))
        
        return combined

# Test functions
def test_burst_integration():
    """Test burst detection integration"""
    print("🧪 Testing Burst Detection Integration")
    print("=" * 50)
    
    # Test the integration pipeline
    pipeline = BurstDetectionPipeline(sample_rate=1e6)
    
    # Test with synthetic signal
    print("Generating test signal...")
    
    # Create a signal with multiple bursts
    total_samples = 50000
    signal = 0.05 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
    
    # Add several burst signals
    burst_positions = [5000, 15000, 25000, 35000, 45000]
    
    for pos in burst_positions:
        if pos + 1000 < total_samples:
            # Create QPSK burst
            burst_data = np.random.randint(0, 2, 100)
            symbols = []
            
            for i in range(0, len(burst_data), 2):
                if i + 1 < len(burst_data):
                    i_bit, q_bit = burst_data[i], burst_data[i+1]
                    symbol = (2*i_bit - 1) + 1j*(2*q_bit - 1)
                    symbols.append(symbol)
            
            # Upsample and add to signal
            upsampled = np.repeat(symbols, 10)
            if len(upsampled) <= 1000:
                signal[pos:pos+len(upsampled)] += upsampled * 3  # Strong signal
    
    print(f"Processing {len(signal)} samples...")
    
    # Start processing
    pipeline.start_processing()
    
    # Process in chunks
    chunk_size = 5000
    total_frames = 0
    
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        frames = pipeline.process_signal_chunk(chunk)
        total_frames += len(frames)
        
        if frames:
            print(f"Chunk {i//chunk_size + 1}: {len(frames)} frames detected")
    
    # Get final statistics
    stats = pipeline.get_processing_statistics()
    
    print(f"\n📊 Final Results:")
    print(f"Total samples processed: {stats['total_samples']:,}")
    print(f"Total frames detected: {total_frames}")
    print(f"Processing time: {stats['processing_time']:.3f} seconds")
    
    if stats['total_samples'] > 0:
        rate = stats['total_samples'] / stats['processing_time']
        print(f"Processing rate: {rate/1e6:.2f} MS/s")
    
    # Stop processing
    pipeline.stop_processing()
    
    print("\n✅ Integration test completed")

if __name__ == "__main__":
    if BURST_DETECTOR_AVAILABLE:
        test_burst_integration()
    else:
        print("❌ Burst detector not available for testing")
'''

# Write the integration module
with open('burst_detection_integration.py', 'w', encoding='utf-8') as f:
    f.write(integration_code)

print("✅ Created burst_detection_integration.py")
print("📁 File size:", len(integration_code), "characters")