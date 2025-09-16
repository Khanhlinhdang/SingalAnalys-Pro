# SDR DETECTION MODULE - PHÂN TÍCH CHI TIẾT
*Giải thích code và tính năng của `sdr._detection` - Module Detection trong thư viện SDR*

## 📊 TỔNG QUAN MODULE

Module `sdr._detection` là một **subpackage chuyên về thuật toán detection** trong xử lý tín hiệu radio. Nó cung cấp các detector algorithms để **phát hiện sự hiện diện của tín hiệu** trong môi trường nhiễu.

### 🗂️ Cấu trúc Module
```
sdr/_detection/
├── __init__.py          # Import tất cả components
├── _correlator.py       # Correlation-based detectors  
├── _energy.py          # Energy detection algorithms
├── _theory.py          # Theoretical performance calculations
└── __pycache__/        # Compiled Python bytecode
```

## 🔍 PHÂN TÍCH CHI TIẾT TỪNG COMPONENT

### 1. **`_correlator.py` - Replica Correlator Detector**

#### 📋 Mục đích:
Implements **clairvoyant replica-correlator detector** - một detector biết trước hoàn toàn về tín hiệu cần detect.

#### 🧮 Cơ sở toán học:
```
Null Hypothesis (H₀):     x[n] = w[n]              (chỉ có noise)
Alternative Hypothesis (H₁): x[n] = s[n] + w[n]     (có signal + noise)

Test Statistic: T(x) = Re(Σ x[n]s*[n]) > γ'

Distribution:
- Under H₀: T(x) ~ N(0, σ²ℰ/2)
- Under H₁: T(x) ~ N(ℰ, σ²ℰ/2)
```

#### 🔧 Tính năng chính:
- **`roc()`**: Tính toán ROC curve (Receiver Operating Characteristic)
- **`p_d()`**: Tính xác suất phát hiện (Probability of Detection)
- **`p_fa()`**: Tính xác suất báo động giả (Probability of False Alarm)

#### 💡 Ứng dụng:
- **Optimal detector** khi biết chính xác signal template
- **Matched filtering** trong communications
- **Radar signal detection** với known waveform

### 2. **`_energy.py` - Energy Detector**

#### 📋 Mục đích:  
Implements **energy detector** - phát hiện tín hiệu dựa trên **năng lượng tổng**.

#### 🧮 Cơ sở toán học:
```
Test Statistic: T(x) = Σ|x[n]|² > γ'

Chi-squared Distribution:
- Under H₀: T(x)/(σ²/2) ~ χ²₂ₙ
- Under H₁: T(x)/((σₛ² + σ²)/2) ~ χ²₂ₙ

Detection Performance:
P_D = Q_χ²₂ₙ(Q⁻¹_χ²₂ₙ(P_FA)/(σₛ²/σ² + 1))
```

#### 🔧 Tính năng chính:
- **`roc()`**: ROC curves cho energy detection
- **`p_d()`**: Probability of detection calculations  
- **`p_fa()`**: False alarm probability
- **`threshold()`**: Optimal threshold selection
- **Non-coherent integration**: Tích hợp nhiều samples

#### 💡 Ứng dụng:
- **Spectrum sensing** trong cognitive radio
- **Signal presence detection** khi không biết signal structure
- **Interference detection** trong wireless systems
- **OFDM signal detection**

### 3. **`_theory.py` - Theoretical Performance**

#### 📋 Mục đích:
Cung cấp **theoretical calculations** cho detection performance.

#### 🔧 Tính năng chính:

##### **`albersheim()` Function:**
Estimates minimum required **single-sample SNR** để đạt được:
- Desired probability of detection (P_D)
- Desired probability of false alarm (P_FA)  
- Given number of non-coherent combinations (N_NC)

#### 🧮 Albersheim's Equation:
```
A = ln(0.62/P_FA)
B = ln(P_D/(1-P_D))

SNR_dB = -5×log₁₀(N_NC) + (6.2 + 4.54/√(N_NC + 0.44)) × 
         log₁₀(A + 0.12×A×B + 1.7×B)
```

#### 💡 Ứng dụng:
- **System design**: Tính SNR requirements
- **Link budget calculations** trong communications
- **Radar system analysis**
- **Performance prediction** cho detection systems

## 🎯 CÁC KHÁI NIỆM DETECTION THEORY

### Detection Performance Metrics:

1. **Probability of Detection (P_D)**:
   - Xác suất detect đúng khi có signal
   - P_D = P(decide H₁ | H₁ is true)

2. **Probability of False Alarm (P_FA)**:
   - Xác suất detect sai khi chỉ có noise
   - P_FA = P(decide H₁ | H₀ is true)

3. **ROC Curve**:
   - Plot P_D vs P_FA
   - Đánh giá performance của detector

4. **Signal-to-Noise Ratio (SNR)**:
   - σₛ²/σ² (signal power / noise power)
   - Determines detection capability

## 🚀 CODE EXAMPLES VÀ SỬ DỤNG

### 1. **Energy Detection Example:**
```python
import sdr
import numpy as np

# Parameters
snr_db = 10          # SNR in dB
N_nc = 25           # Number of samples to integrate
p_fa = 1e-6         # Desired false alarm probability

# Compute ROC curve
p_fa_range, p_d = sdr.EnergyDetector.roc(snr_db, N_nc)

# Compute probability of detection for specific P_FA
p_d_specific = sdr.EnergyDetector.p_d(snr_db, N_nc, p_fa)

print(f"P_D for SNR={snr_db}dB, N_nc={N_nc}, P_FA={p_fa}: {p_d_specific:.4f}")
```

### 2. **Replica Correlator Example:**
```python
# Energy-to-noise ratio
enr_db = 15         # ENR in dB
p_fa = 1e-5         # False alarm probability

# Compute detection probability
p_d = sdr.ReplicaCorrelator.p_d(enr_db, p_fa, complex=True)

print(f"P_D for ENR={enr_db}dB, P_FA={p_fa}: {p_d:.4f}")
```

### 3. **Albersheim SNR Estimation:**
```python
# Design requirements
p_d_required = 0.9      # 90% detection probability
p_fa_required = 1e-6    # 1 in million false alarms
N_nc = 10              # 10 non-coherent integrations

# Calculate minimum required SNR
min_snr = sdr.albersheim(p_d_required, p_fa_required, N_nc)

print(f"Minimum SNR required: {min_snr:.2f} dB")
```

## 📊 APPLICATIONS TRONG RF SPECTRUM ANALYZER

### 1. **Signal Presence Detection:**
```python
def detect_signal_presence(iq_samples, noise_variance):
    """Detect if signal is present in IQ samples."""
    
    # Calculate energy
    energy = np.sum(np.abs(iq_samples)**2)
    
    # Set detection threshold
    N_samples = len(iq_samples)
    p_fa_target = 1e-6
    threshold = sdr.EnergyDetector.threshold(
        N_samples, p_fa_target, noise_variance, complex=True
    )
    
    # Make detection decision
    signal_detected = energy > threshold
    
    return signal_detected, energy, threshold
```

### 2. **Automatic Gain Control (AGC) Trigger:**
```python
def agc_trigger_detection(signal, target_p_fa=1e-5):
    """Use energy detection to trigger AGC."""
    
    # Estimate noise floor
    noise_est = np.var(signal[:1000])  # Use first 1000 samples
    
    # Energy detection
    energy = np.sum(np.abs(signal)**2)
    N = len(signal)
    
    # Calculate threshold
    threshold = sdr.EnergyDetector.threshold(N, target_p_fa, noise_est)
    
    # Trigger AGC if strong signal detected
    return energy > threshold * 10  # 10x threshold for AGC trigger
```

### 3. **Spectrum Sensing for Cognitive Radio:**
```python
def spectrum_sensing(iq_data, freq_bands, p_fa=1e-4):
    """Perform spectrum sensing across frequency bands."""
    
    detection_results = {}
    
    for band_name, (f_start, f_stop) in freq_bands.items():
        # Filter to frequency band
        band_signal = bandpass_filter(iq_data, f_start, f_stop)
        
        # Energy detection
        energy = np.sum(np.abs(band_signal)**2)
        N = len(band_signal)
        
        # Estimate noise variance for this band
        noise_var = estimate_noise_variance(band_signal)
        
        # Detection threshold
        threshold = sdr.EnergyDetector.threshold(N, p_fa, noise_var)
        
        # Detection decision
        detection_results[band_name] = {
            'detected': energy > threshold,
            'energy': energy,
            'snr_est': 10*np.log10(energy/noise_var/N) if noise_var > 0 else -np.inf
        }
    
    return detection_results
```

## 🎯 INTEGRATION VỚI RF SPECTRUM ANALYZER

### Trong RF Spectrum Analyzer project, `sdr._detection` có thể được sử dụng để:

1. **Signal Detection trong Modulation Analysis:**
   - Xác định có signal hay chỉ noise trước khi analyze modulation
   - Set adaptive thresholds cho modulation detection

2. **Automatic Signal Classification:**
   - Energy-based pre-filtering trước khi chạy complex algorithms
   - Confidence scoring cho modulation detection results

3. **Performance Monitoring:**
   - Calculate theoretical vs actual detection performance
   - Validate detector performance với known test signals

4. **Adaptive Processing:**
   - Adjust processing parameters dựa trên detected signal strength
   - Optimize resource allocation cho strong vs weak signals

## 🏆 TÓM TẮT

**`sdr._detection`** module cung cấp:

✅ **Complete detection framework** với theoretical foundations
✅ **Energy detector** cho general signal detection  
✅ **Replica correlator** cho optimal known-signal detection
✅ **Performance prediction tools** (Albersheim's equation)
✅ **ROC analysis capabilities** cho system design
✅ **Chi-squared và Gaussian statistics** implementations

**Đây là foundation module quan trọng** cho bất kỳ RF system nào cần:
- Signal presence detection
- Performance analysis và optimization  
- Theoretical validation của detection algorithms
- Adaptive signal processing

Module này **complement** rất tốt với RF Spectrum Analyzer để cung cấp **robust signal detection capabilities** trước khi chạy các advanced modulation analysis algorithms.

---
*Analysis completed: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")*
*SDR Detection Module - Professional RF Signal Processing Library*