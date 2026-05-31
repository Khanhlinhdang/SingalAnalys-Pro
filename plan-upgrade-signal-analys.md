# Kế Hoạch Nâng Cấp Signal Analysis

## Mục tiêu tổng thể
Xây dựng RF Spectrum Analyzer thành nền tảng phát hiện và phân tích tín hiệu vệ tinh (Inmarsat/Iridium/NOAA...) theo pipeline đầy đủ từ RF input đến kết quả cụ thể (text/voice/image khi có protocol handler), có khả năng kiểm chứng từng stage.

## Nguyên tắc thực hiện
- Ưu tiên khả năng vận hành thời gian thực và đo được hiệu năng.
- Mỗi phase đều có acceptance criteria và test plan rõ ràng.
- Triển khai theo từng increment nhỏ, không phá vỡ luồng hiện có (SpyServer, demo mode, file source).

## Full Pipeline Cần Đạt (Input -> Kết Quả)
1. RF Capture Input:
  - SDR backend thu complex IQ samples từ SpyServer/SDR source.
2. Ingest and Buffering:
  - Đưa IQ chunks vào bounded/ring buffers để đảm bảo stream ổn định và không nghẽn GUI.
3. Preprocessing:
  - Chuẩn hóa mức tín hiệu, windowing, optional filtering, frame selection.
4. Detection Stage:
  - Signal/noise decision, energy detection, burst detection.
5. Auto Characterization Stage:
  - Ước lượng modulation family và các tham số chính (symbol rate, occupied bandwidth, clock hints).
6. Demodulation Stage:
  - Phục hồi symbols/bit candidates.
  - Bắt buộc có constellation view để xác nhận chất lượng và độ đúng của demod.
7. Dechannelization / Demultiplexing Stage:
  - Tách luồng kênh/stream khi có multiplexing.
8. Deinterleaving / Descrambling Stage:
  - Đảo interleaver/scrambler nếu phát hiện được.
9. FEC Decoding Stage:
  - Giải mã channel coding để thu payload bits.
10. Payload Parsing Stage:
   - Chuyển payload bits thành dữ liệu có cấu trúc theo protocol.
11. Output Rendering Stage:
   - Hiển thị kết quả text, bitstream, constellation, và application-specific outputs (audio/video/image) khi có handler.

## Tiêu chí thành công toàn tuyến
- Từ 1 ROI hợp lệ có thể đi qua đầy đủ các stage tới output cụ thể (ít nhất text hoặc voice/audio tùy protocol).
- Có telemetry và trạng thái theo stage để biết pipeline đúng ở đâu, fail ở đâu.
- Constellation cho phép đối chiếu nhanh với modulation dự kiến để xác nhận demod đúng.

## Phase 1 - Nền ROI Processing & Telemetry
### Mục đích
Chuẩn hóa đường đi dữ liệu ROI trong toàn bộ pipeline và bổ sung telemetry để xác nhận ROI extraction/decimation đang chạy đúng như thiết kế, tạo nền tảng cho full pipeline.

### Mục tiêu cần đạt
- Kết quả phân tích phải kèm theo metadata runtime cho ROI:
  - roi_freq_start_hz, roi_freq_end_hz, roi_bandwidth_hz
  - source_sample_rate_hz, analysis_sample_rate_hz
  - decimation_factor, iq_samples_used
- GUI hiển thị được thông tin cơ bản để người dùng kiểm chứng nhanh.
- Không làm giảm FPS theo cảm nhận khi phân tích ROI hẹp.
- Có stage tagging cơ bản (capture/ingest/preprocess/detect) trong telemetry để chuẩn bị cho pipeline đầy đủ.

### Deliverables
- Bổ sung analysis context metadata trong flow signal analysis request (sync + async).
- Cập nhật GUI info text có thông tin sample-rate/decimation.
- Logging bổ sung cho telemetry ROI để debug và benchmark.
- Định nghĩa schema chung cho stage status (pending/running/success/failed) để sử dụng xuyên suốt các phase sau.

### Phương án test
- Unit-level:
  - Kiểm tra payload analysis có đầy đủ key metadata khi có freq_range.
  - Kiểm tra decimation_factor >= 1 và hợp lý với ROI hẹp.
  - Kiểm tra stage status schema có đầy đủ field bắt buộc.
- Integration-level:
  - Chọn ROI rộng và ROI hẹp trên spectrum, bấm Analyze Signal.
  - So sánh analysis_sample_rate_hz và decimation_factor giữa 2 trường hợp.
- Runtime smoke:
  - Chạy app với SpyServer và demo mode, đảm bảo không crash, GUI update bình thường.

### Acceptance criteria
- Có metadata trong kết quả phân tích cho cả sync và async path.
- GUI hiển thị thông tin modulation + SNR + telemetry cơ bản (sample rate/decimation).
- Không phát sinh warning mới trong file đã sửa.
- Stage telemetry schema sử dụng được trong flow hiện tại.

## Phase 2 - Multi-ROI & Channel Jobs
### Mục đích
Cho phép quản lý nhiều ROI và chạy phân tích theo job queue để phục vụ scan/compare nhiều kênh.

### Mục tiêu cần đạt
- Hỗ trợ tạo/sửa/xóa danh sách ROI presets.
- Có cơ chế enqueue/debounce phân tích, tránh spam request.
- Mỗi ROI trả về kết quả riêng, có timestamp và độ ưu tiên.
- Tạo channel job abstraction để tiếp tục sang các stage demod/decode.

### Deliverables
- Data model ROI list + persistence trong settings.
- Scheduler job phân tích ROI.
- Bảng/tile hiển thị kết quả theo ROI.
- Job result envelope có stage-status + stage-metrics.

### Phương án test
- Unit-level: CRUD ROI, validate clamp/frequency ordering.
- Integration-level: queue 3-5 ROI, xác minh kết quả map đúng ROI.
- Performance-level: đo thời gian xử lý trung bình mỗi ROI.

### Acceptance criteria
- Người dùng có thể trigger phân tích nhiều ROI liên tiếp không race/crash.
- Kết quả không bị ghi đè sai ROI.
- Stage metrics theo ROI hiển thị đúng.

## Phase 3 - Detection và Auto Characterization Đầy Đủ
### Mục đích
Hoàn thiện detect + auto characterization để đưa dữ liệu vào demod với tham số hợp lý.

### Mục tiêu cần đạt
- Nâng cao signal/noise decision, burst segmentation.
- Ước lượng modulation family và key parameters có confidence.
- Sinh được đề xuất demod mode cho stage tiếp theo.

### Deliverables
- Detection module nâng cấp (energy + burst + optional cyclostationary heuristics nếu khả thi).
- Characterization module trả về modulation candidates + confidence.
- GUI panel hiển thị kết quả characterization.

### Phương án test
- Unit test với synthetic IQ (BPSK/QPSK/FSK/OFDM-like mock).
- Integration test với file mẫu và live ROI.
- Đo confusion matrix modulation family trên bộ test có nhãn.

### Acceptance criteria
- Có modulation candidate + confidence ổn định cho các case có tín hiệu.
- Không làm vỡ compatibility của flow phân tích cũ.

## Phase 4 - Demodulation + Constellation Verification
### Mục đích
Xây dựng demod stage đủ tin cậy và có khả năng xác nhận bằng constellation.

### Mục tiêu cần đạt
- Recover symbols/bit candidates cho các modulation mục tiêu.
- Hiện constellation theo frame, có chỉ số EVM/cluster spread cơ bản (nếu khả thi).
- Có cơ chế so khớp constellation pattern để cảnh báo demod sai.

### Deliverables
- Demod engine profile-based.
- Constellation verification block + UI overlay.
- Bit candidate stream output cho phase sau.

### Phương án test
- Unit test bởi dữ liệu tổng hợp cho từng modulation.
- Visual verification constellation với test vectors.
- Regression test hiệu năng để đảm bảo GUI không giật.

### Acceptance criteria
- Constellation view đồng bộ với kết quả demod.
- Có bằng chứng demod đúng qua BER/decision metric trên bộ test.

## Phase 5 - Dechannelization / Demultiplexing
### Mục đích
Tách các channel/stream khi tín hiệu có multiplexing.

### Mục tiêu cần đạt
- Có pipeline tách stream cho ít nhất 1 kiểu multiplexing mục tiêu.
- Mỗi stream con được đưa tiếp vào deinterleave/descramble/FEC.

### Deliverables
- Dechannelization module + stream router.
- Stream metadata và mapping ra output panel.

### Phương án test
- Unit test demux logic với frame giả lập.
- Integration test với sample có multiplexing.

### Acceptance criteria
- Số stream tách ra đúng theo kỳ vọng trên bộ test.

## Phase 6 - Deinterleaving / Descrambling
### Mục đích
Khôi phục thứ tự/bit pattern gốc trước khi vào FEC.

### Mục tiêu cần đạt
- Phát hiện/áp dụng được interleaver/scrambler profile-supported.

### Deliverables
- Deinterleaver + descrambler modules.
- Heuristic chọn profile nếu có nhiều candidates.

### Phương án test
- Unit test với vectors đã biết kết quả.
- Integration test trên stream sau demux.

### Acceptance criteria
- Tỷ lệ lỗi bit trước FEC giảm rõ rệt sau stage này.

## Phase 7 - FEC Decoding
### Mục đích
Giải mã coding để thu payload bits ổn định.

### Mục tiêu cần đạt
- Hỗ trợ ít nhất 1-2 FEC profiles ưu tiên theo target.

### Deliverables
- FEC decoder block + metrics (pre/post BER, decode success).

### Phương án test
- Unit test với codewords chuẩn.
- Integration test với dữ liệu có nhiều mức SNR.

### Acceptance criteria
- Decode thành công trên bộ test baseline theo ngưỡng BER đặt ra.

## Phase 8 - Payload Parsing và Output Rendering
### Mục đích
Chuyển bits thành kết quả người dùng có thể sử dụng trực tiếp.

### Mục tiêu cần đạt
- Parse payload thành schema có cấu trúc.
- Render được text, bitstream, constellation, và audio/voice (nếu có handler).

### Deliverables
- Protocol handlers ưu tiên (theo target mission).
- Output renderer cho text/log + media outputs.
- Export kết quả phân tích theo session.

### Phương án test
- Golden file tests cho parser.
- End-to-end test từ ROI đến output cụ thể.
- User validation test: đối chiếu kết quả với dữ liệu tham chiếu.

### Acceptance criteria
- Có ít nhất 1 luồng end-to-end trả về kết quả cụ thể (text hoặc voice).
- Lỗi parser/handler được báo cáo rõ ràng, không fail im lặng.

## Phase 9 - Operational UX & Observability
### Mục đích
Đảm bảo hệ thống vận hành được dài hạn, dễ giám sát và dễ debug.

### Mục tiêu cần đạt
- Dashboard metrics (FPS, queue depth, latency, drop rate, stage success ratio).
- Mission presets workflow và session reports.

### Deliverables
- Runtime metrics panel.
- Preset manager.
- Session report exporter (json/markdown).

### Phương án test
- UI test luồng preset.
- Soak test 15-30 phút với SpyServer.
- Verify report schema/nội dung.

### Acceptance criteria
- Dashboard cập nhật liên tục, không lag giao diện rõ rệt.
- Preset hoạt động qua restart app.

## Kế hoạch triển khai ngay (bắt đầu trong lần này)
1. Thực hiện Phase 1 - Deliverable đầu tiên:
   - Thêm analysis context metadata vào kết quả phân tích (sync + async).
   - Hiển thị telemetry cơ bản trên info label.
2. Chạy kiểm tra syntax/lint errors cho file sửa.
3. Chạy smoke test nhanh để xác nhận luồng phân tích không vỡ.
4. Tiếp tục increment tiếp theo:
  - Bổ sung stage-tag schema và stage-status envelope.
  - Chuẩn bị giao diện hiển thị kết quả pipeline theo stage.

## Checklist theo dõi tiến trình

### Quy ước trạng thái
- [ ] Chưa bắt đầu
- [~] Đang thực hiện
- [x] Hoàn thành

### Checklist tổng quan theo phase
- [x] Phase 1 - Nền ROI Processing & Telemetry
  - [x] Bổ sung analysis context metadata (sync + async)
  - [x] Hiển thị telemetry cơ bản trên GUI info label
  - [x] Kiểm tra syntax/language errors cho file đã sửa
  - [x] Hoàn thiện stage-tag schema (capture/ingest/preprocess/detect)
  - [x] Chạy smoke-check nhanh bằng test suite `--fast`
  - [x] Chạy runtime smoke test GUI với SpyServer
  - [x] Chạy runtime smoke test GUI với demo mode
- [~] Phase 2 - Multi-ROI & Channel Jobs
  - [x] Tạo nền queue ROI requests trong app state
  - [x] Thêm debounce cho request ROI trùng lặp tức thời
  - [x] Thiết kế ROI presets persistence trong settings
  - [x] Bổ sung panel kết quả/queue theo ROI trong GUI
  - [x] Nối trạng thái queue ROI (queued/running/completed/failed) vào panel
  - [~] Chuẩn hóa job result envelope theo ROI cho các phase sau
- [ ] Phase 3 - Detection và Auto Characterization Đầy Đủ
- [ ] Phase 4 - Demodulation + Constellation Verification
- [ ] Phase 5 - Dechannelization / Demultiplexing
- [ ] Phase 6 - Deinterleaving / Descrambling
- [ ] Phase 7 - FEC Decoding
- [ ] Phase 8 - Payload Parsing và Output Rendering
- [ ] Phase 9 - Operational UX & Observability

### Checklist đầu việc ngắn hạn (tuần hiện tại)
- [x] Chạy smoke test với SpyServer sau patch telemetry
- [x] Chạy smoke test với demo mode sau patch telemetry
- [x] Chốt schema stage-status envelope dùng chung
- [~] Thiết kế panel hiển thị trạng thái theo stage
- [x] Thiết kế data model ROI presets + persistence
- [x] Nối queue ROI với bảng/tile kết quả theo ROI

## Cập nhật tiến trình thực hiện kế hoạch
- Tiến độ tổng quan: 1/9 phase hoàn tất (Phase 1), 1/9 phase đang triển khai (Phase 2).
- Tiến độ kỹ thuật đã hoàn thành:
  - Đã có metadata ROI/runtime trong luồng phân tích.
  - Đã hiển thị telemetry cơ bản (sample rate/decimation) trên GUI.
  - Đã bổ sung stage-status envelope chuẩn hóa trong `analysis_context`.
  - Đã xác nhận schema stage-status bằng chạy snippet kiểm tra key bắt buộc.
  - Đã xác nhận không có lỗi cú pháp ở file chính đã chỉnh sửa.
  - Đã chạy test suite nhanh `rf_spectrum_analyzer/tests/run_tests.py --fast` (đa số module PASS; còn lỗi môi trường/phần plugin ngoài phạm vi Phase 1).
  - Đã chạy smoke test GUI ngắn hạn cho cả demo mode và SpyServer, ứng dụng khởi tạo thành công.
  - Đã thêm queue + debounce cho ROI analysis request làm nền cho Phase 2.
  - Đã bổ sung persistence ROI presets vào settings (lưu YAML, có `last_selected_preset`, giới hạn số preset gần nhất).
  - Đã thêm ROI Analysis Queue panel trên GUI và nối dữ liệu queue để hiển thị trạng thái theo ROI.
  - Đã nối cập nhật trạng thái ROI request theo vòng đời phân tích: queued -> running -> completed/failed/skipped.
- Việc còn mở ngay lúc này:
  - Cần hoàn thiện schema job result envelope theo ROI để dùng ổn định cho Phase 3+.
  - Cần bổ sung test tích hợp queue nhiều ROI liên tiếp để đo độ ổn định mapping kết quả.