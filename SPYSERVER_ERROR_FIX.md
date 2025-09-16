# SpyServer Connection Error Fix Documentation

## 🔧 Problem Fixed

**Error**: `[WinError 10061] No connection could be made because the target machine actively refused it`

**Root Cause**: SpyServer software không chạy hoặc không accessible tại địa chỉ đã cấu hình.

## ✅ Solutions Implemented

### 1. Improved Error Handling
- **File**: `rf_spectrum_analyzer/backends/spyserver_backend.py`
- **Changes**:
  - Graceful handling cho `ConnectionRefusedError`
  - Specific error messages cho từng loại lỗi
  - User-friendly instructions
  - Warning level thay vì error level cho expected failures

```python
# Before (ERROR level):
except SDRConnectError as e:
    logger.error(f"SpyServer connection error: {e}")
    return False

# After (WARNING level with helpful info):
except ConnectionRefusedError as e:
    logger.warning(f"SpyServer connection refused at {self.host}:{self.port}")
    logger.info("SpyServer is not running or accessible. Please start SpyServer or check connection settings.")
    return False
```

### 2. Auto Demo Mode Fallback
- **File**: `rf_spectrum_analyzer/core/app.py`
- **Changes**:
  - Auto-detect SDR connection failures
  - Automatic fallback to demo mode
  - Graceful degradation

```python
def _try_sdr_connection(self):
    """Try to connect to SDR device, enable demo mode if connection fails."""
    try:
        if self.sdr_manager.connect():
            self.logger.info(f"Successfully connected to {device_type}")
            self.demo_mode = False
        else:
            self.logger.warning(f"Failed to connect to {device_type}, enabling demo mode")
            self._enable_demo_mode()
    except Exception as e:
        self.logger.warning(f"SDR connection error: {e}")
        self._enable_demo_mode()
```

### 3. Command Line Demo Mode
- **File**: `main.py`
- **Changes**:
  - Added `--demo` flag
  - Force demo mode option

```bash
# Run with demo mode
python main.py --demo

# Run with specific device
python main.py --device spyserver

# Run with debug info
python main.py --debug
```

## 🎯 Usage Scenarios

### Scenario 1: No SpyServer Available
```bash
python main.py --device spyserver
```
**Result**: App detects connection failure → automatically enables demo mode → continues working

### Scenario 2: Force Demo Mode
```bash
python main.py --demo
```
**Result**: App runs in demo mode từ đầu, không cố gắng connect SDR

### Scenario 3: SpyServer Available
```bash
python main.py --device spyserver
```
**Result**: App connects successfully → normal operation với real data

## 🔍 Error Messages Explained

### Old Error Message (Confusing):
```
ERROR - SpyServer connection error: Failed to connect to SpyServer: [WinError 10061] No connection could be made because the target machine actively refused it
```

### New Error Messages (Clear):
```
WARNING - SpyServer connection refused at localhost:5555
INFO - SpyServer is not running or accessible. Please start SpyServer or check connection settings.
INFO - To use SpyServer: 1) Start SpyServer software, 2) Verify host/port settings, 3) Check firewall
```

## 🛠️ How to Use SpyServer

### Step 1: Install SpyServer Software
1. Download SpyServer từ official source
2. Install và configure cho SDR hardware của bạn
3. Start SpyServer application

### Step 2: Configure RF Spectrum Analyzer
```python
# In GUI or settings
settings.sdr.spyserver_host = "your.spyserver.host"
settings.sdr.spyserver_port = 5555
settings.sdr.spyserver_timeout = 10.0
```

### Step 3: Connect
```bash
python main.py --device spyserver
```

## 🧪 Testing the Fix

### Test Script Available:
```bash
python test_spyserver_error_handling.py
```

### Expected Output:
```
✓ Connection failed as expected (no SpyServer running)
✓ Error handling worked correctly
✅ All tests passed!
```

## 📋 Files Modified

1. **`rf_spectrum_analyzer/backends/spyserver_backend.py`**
   - Improved error handling
   - User-friendly messages
   - Connection troubleshooting tips

2. **`rf_spectrum_analyzer/core/app.py`**
   - Auto-fallback to demo mode
   - Graceful SDR connection handling

3. **`main.py`**
   - Added `--demo` command line option
   - Enhanced argument parsing

4. **`test_spyserver_error_handling.py`** (New)
   - Test script cho error handling
   - Verification tool

## 🎉 Benefits

1. **Better User Experience**: Không crash khi SpyServer không available
2. **Clear Instructions**: User biết chính xác cần làm gì
3. **Automatic Fallback**: App vẫn hoạt động với demo data
4. **Easy Testing**: Demo mode cho development và testing
5. **Graceful Degradation**: App handles failures elegantly

## 🔄 Before vs After

### Before:
- App shows scary ERROR messages
- User confused về cách fix
- No guidance về SpyServer setup
- Hard to test without real hardware

### After:
- Clear WARNING messages với instructions
- Auto-fallback to working demo mode
- Step-by-step guidance cho SpyServer setup
- Easy testing với `--demo` flag
- Graceful degradation