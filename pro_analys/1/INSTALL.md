# Hướng dẫn cài đặt phần mềm SDR

## Yêu cầu hệ thống

### Phần cứng
- USRP N210 hoặc X310
- Máy tính với cổng Ethernet (cho N210) hoặc SFP+/10GbE (cho X310)
- RAM: tối thiểu 4GB, khuyến nghị 8GB+
- CPU: Intel i5 hoặc tương đương trở lên

### Phần mềm
- Ubuntu 20.04/22.04 LTS hoặc Windows 10/11
- Python 3.8 trở lên
- UHD (USRP Hardware Driver) 4.0+

## Cài đặt chi tiết

### 1. Cài đặt UHD

#### Ubuntu/Linux:
```bash
# Cài đặt dependencies
sudo apt update
sudo apt install libuhd-dev libuhd4.1.0 uhd-host

# Download FPGA images
sudo uhd_images_downloader

# Kiểm tra kết nối USRP
uhd_find_devices
```

#### Windows:
- Tải UHD từ: https://files.ettus.com/binaries/uhd/
- Cài đặt UHD installer
- Thêm UHD bin directory vào PATH
- Download FPGA images bằng uhd_images_downloader.exe

### 2. Cài đặt Python dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv sdr_env
source sdr_env/bin/activate  # Linux/Mac
# hoặc sdr_env\Scriptsctivate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### 3. Cấu hình mạng (cho USRP N210)

#### Linux:
```bash
# Cấu hình static IP cho interface kết nối USRP
sudo ip addr add 192.168.10.1/24 dev eth0
sudo ip link set eth0 up

# Hoặc sửa /etc/netplan/01-netcfg.yaml:
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 192.168.10.1/24
```

#### Windows:
- Mở Network Connections
- Chuột phải vào Ethernet adapter kết nối USRP
- Properties → TCP/IPv4 → Properties
- Chọn "Use the following IP address"
- IP: 192.168.10.1, Subnet: 255.255.255.0

### 4. Kiểm tra kết nối

```bash
# Test ping tới USRP (default IP: 192.168.10.2)
ping 192.168.10.2

# Test UHD connection
uhd_find_devices --args="addr=192.168.10.2"

# Test Python UHD bindings
python -c "import uhd; print('UHD version:', uhd.get_version_string())"
```

## Chạy phần mềm

```bash
python sdr_application.py
```

## Cấu hình USRP

### USRP N210
- Default IP: 192.168.10.2
- Device args: "addr=192.168.10.2" hoặc "type=usrp2"
- Max sample rate: 25 MS/s
- Frequency range: DC - 6 GHz (tùy daughterboard)

### USRP X310  
- Default IP: 192.168.10.2 (1GbE), 192.168.40.2 (10GbE)
- Device args: "addr=192.168.10.2" hoặc "type=x300"
- Max sample rate: 200 MS/s
- Frequency range: DC - 6 GHz (tùy daughterboard)

## Troubleshooting

### Lỗi "No UHD devices found"
- Kiểm tra kết nối mạng
- Kiểm tra IP configuration
- Thử: `sudo uhd_find_devices`
- Kiểm tra firewall settings

### Lỗi "Runtime error: Expected FPGA image..."
- Chạy: `sudo uhd_images_downloader`
- Kiểm tra FPGA image path trong UHD

### Lỗi Python import
- Kiểm tra virtual environment activated
- Cài đặt lại packages: `pip install -r requirements.txt`
- Kiểm tra Python version compatibility

### Performance issues
- Tăng buffer size: `net.core.rmem_max`, `net.core.wmem_max`
- Sử dụng SSD thay vì HDD
- Đóng các ứng dụng không cần thiết
- Sử dụng realtime kernel (Linux)

## Tính năng nâng cao

### Multi-threading optimization
```python
# Cấu hình trong code
self.usrp_interface.set_num_recv_frames(32)
self.usrp_interface.set_recv_timeout(0.1)
```

### Ghi dữ liệu hiệu suất cao
```python
# Sử dụng binary format cho IQ data
with open('data.bin', 'wb') as f:
    iq_interleaved = np.column_stack((np.real(iq_data), np.imag(iq_data)))
    f.write(iq_interleaved.astype(np.float32).tobytes())
```

## Ví dụ sử dụng

### Kết nối và thu tín hiệu
1. Khởi động ứng dụng
2. Nhập Device Args (vd: "addr=192.168.10.2")  
3. Click "Connect"
4. Cài đặt tần số, sample rate, gain
5. Quan sát spectrum và waterfall
6. Chọn loại demodulation
7. Xem constellation diagram

### Quét phổ tần số
1. Cài đặt Start Freq và End Freq
2. Click "Start Scan"
3. Quan sát detected signals trong spectrum plot

### Ghi dữ liệu IQ
1. Nhập filename (vd: "recording.bin")
2. Click "Start Recording"
3. Dữ liệu được ghi dưới dạng interleaved I/Q float32

## Liên hệ hỗ trợ

- Ettus Research support: https://kb.ettus.com/
- UHD documentation: https://files.ettus.com/manual/
- GNU Radio community: https://www.gnuradio.org/
