# 🔬 RF Spectrum Analyzer - Bitstream Dock Feature

## Tổng quan

Tính năng **Bitstream Dock** mới đã được thêm vào RF Spectrum Analyzer, cho phép hiển thị và phân tích bitstream theo thời gian thực với khả năng docker tương tự như Constellation Display.

## ✨ Tính năng chính

### 🖼️ Hiển thị Bitstream trực quan
- **Hiển thị pixel màu**: Bit 1 = màu xanh lá, Bit 0 = màu đen
- **Layout linh hoạt**: Có thể điều chỉnh số bit mỗi hàng (16-128 bit)
- **Kích thước pixel**: Có thể thay đổi từ 3x3 đến 15x15 pixel
- **Auto-scroll**: Tự động cuộn xuống khi có dữ liệu mới

### 📊 Phân tích thống kê
- **Entropy calculation**: Tính toán entropy Shannon để đánh giá độ ngẫu nhiên
- **Bit distribution**: Hiển thị tỷ lệ bit 0 và bit 1
- **Bit rate**: Hiển thị tốc độ nhận dữ liệu (bps/kbps)
- **Buffer usage**: Theo dõi mức sử dụng buffer

### 🚢 Docker functionality
- **Movable**: Có thể di chuyển dock đến các vị trí khác nhau
- **Floatable**: Có thể tách thành cửa sổ riêng biệt
- **Closable**: Có thể ẩn/hiện dock
- **Menu integration**: Tích hợp vào View menu với shortcut **Ctrl+B**

### 🎛️ Điều khiển
- **Pause/Resume**: Tạm dừng và tiếp tục nhận dữ liệu
- **Clear**: Xóa toàn bộ dữ liệu hiển thị
- **Save**: Lưu hình ảnh bitstream ra file PNG
- **Export**: Xuất dữ liệu ở nhiều định dạng (binary, hex, numpy)

## 🏗️ Kiến trúc tích hợp

### Thành phần mới
```
rf_spectrum_analyzer/
├── gui/
│   ├── bitstream_widget.py      # Widget hiển thị bitstream
│   └── main_window.py           # Tích hợp dock functionality
├── core/
│   ├── app.py                   # Data flow và timer management
│   └── signal_processor.py     # Xử lý tín hiệu để tạo bitstream
└── tests/
    ├── test_bitstream_dock.py   # Test dock functionality
    └── demo_bitstream.py        # Demo standalone
```

### Data Flow
```
IQ Samples → Signal Processing → Demodulation → Bitstream → Display Widget
     ↓              ↓                ↓              ↓           ↓
  SDR Device → Modulation → Digital Detection → Binary Data → Visual Display
```

## 📋 Cách sử dụng

### 1. Trong RF Spectrum Analyzer
```bash
# Chạy ứng dụng chính
python main.py --debug
```

- Bitstream dock sẽ xuất hiện ở phía dưới giao diện
- Khi có tín hiệu số được phát hiện, bitstream sẽ hiển thị tự động
- Sử dụng menu **View → Bitstream Display** hoặc **Ctrl+B** để ẩn/hiện

### 2. Demo độc lập
```bash
# Chạy demo bitstream widget
python demo_bitstream.py
```

### 3. Test functionality
```bash
# Chạy test tự động và interactive
python test_bitstream_dock.py
```

## 🎮 Phím tắt

| Phím tắt | Chức năng |
|----------|-----------|
| **Ctrl+B** | Toggle Bitstream Dock |
| **Ctrl+D** | Toggle Constellation Dock |
| **Ctrl+R** | Reset Layout |
| **F11** | Fullscreen |

## 🔧 Cấu hình

### Bitstream Display Settings
- **Bits per row**: 16-128 bit (mặc định: 32)
- **Pixel size**: 3-15 pixel (mặc định: 6)
- **Buffer size**: Tối đa 50,000 bit
- **Update rate**: 10 FPS cho bitstream display

### Color Scheme
- **Bit 0**: `#141414` (Dark gray)
- **Bit 1**: `#00ff64` (Bright green)
- **Background**: `#1a1a1a` (Very dark gray)

## 📊 Patterns hỗ trợ

Bitstream widget có thể hiển thị nhiều loại pattern khác nhau:

1. **Alternating Pattern**: 010101...
2. **Block Pattern**: 111000111000...
3. **Manchester Encoding**: Dual-state encoding
4. **PRBS**: Pseudo-Random Binary Sequence
5. **Sync + Data**: Frame với sync word
6. **Random Data**: High entropy data
7. **Low Entropy**: Sparse bit patterns
8. **Binary Counter**: Sequential counting

## 🧪 Testing

### Automated Tests
```python
# Test dock properties
test_dock_properties()

# Test menu integration  
test_menu_integration()
```

### Interactive Tests
- Toggle visibility
- Float/dock operations
- Layout reset
- Data generation patterns
- Export functionality

## 🎯 Performance

- **Memory efficient**: Buffer giới hạn để tránh memory leak
- **Optimized rendering**: Batch updates và pixel-perfect drawing
- **Configurable refresh**: 10 FPS update rate có thể điều chỉnh
- **Background processing**: Data processing không block GUI

## 🔗 Tích hợp với Signal Processing

Bitstream dock được tích hợp với signal processing pipeline:

1. **Modulation Analysis**: Phát hiện loại modulation
2. **Demodulation**: Tách dữ liệu từ tín hiệu RF
3. **Digital Detection**: Nhận biết tín hiệu số
4. **Binary Conversion**: Chuyển đổi sang bit stream
5. **Real-time Display**: Hiển thị trực tiếp

## 🎨 UI/UX Features

- **Dark theme**: Phù hợp với RF analysis environment
- **Professional layout**: Consistent với constellation dock
- **Responsive controls**: Real-time parameter adjustment
- **Status indicators**: Clear feedback về trạng thái hệ thống
- **Error handling**: Graceful error management

## 🚀 Future Enhancements

Các cải tiến có thể thêm trong tương lai:

- **Protocol Decoding**: Giải mã các protocol cụ thể
- **Pattern Recognition**: Tự động nhận diện pattern
- **Data Logging**: Lưu bitstream vào file log
- **Advanced Analysis**: Phân tích chu kỳ và correlations
- **Custom Color Schemes**: Nhiều bảng màu khác nhau

## 📝 Ví dụ sử dụng

### Phân tích tín hiệu FSK
```python
# Bitstream sẽ hiển thị tự động khi phát hiện FSK signal
# Entropy cao → Random data
# Entropy thấp → Structured data
```

### Monitor Digital Communications
```python
# Quan sát sync patterns và frame structure
# Phát hiện lỗi bit transmission
# Đánh giá chất lượng tín hiệu
```

## 🏆 Kết luận

Tính năng Bitstream Dock mang lại khả năng phân tích bitstream mạnh mẽ cho RF Spectrum Analyzer:

✅ **Hoàn thành**: Docker functionality giống Constellation Display  
✅ **Hoàn thành**: Real-time bitstream visualization  
✅ **Hoàn thành**: Statistical analysis và entropy calculation  
✅ **Hoàn thành**: Interactive controls và export capabilities  
✅ **Hoàn thành**: Integration với signal processing pipeline  

Tính năng này giúp engineers có thể:
- Quan sát bitstream pattern trong thời gian thực
- Phân tích chất lượng tín hiệu số
- Debug digital communication systems
- Nghiên cứu encoding schemes và protocols