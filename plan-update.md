# RF Decode System Upgrade Plan (Codebase-Driven)

## 1. Current Project Assessment (Based on Actual Code)

### 1.1 Maturity Snapshot

- Capture and UI visualization: strong
- Core DSP (detection, modulation analysis, demodulation): moderate to strong
- Coding/FEC decoding: moderate (broad classes, uneven protocol depth)
- Protocol parsing and rich output (text/mail/audio/video): low to moderate
- End-to-end reproducible decode chains per target protocol: low to moderate
- Test coverage for protocol output stages: low

### 1.2 Implemented vs Partial vs Missing

Implemented:
- Multi-backend SDR abstraction and acquisition path
- Processing worker with backpressure and latest-wins queue
- Spectrum/waterfall/constellation rendering
- Baseline modulation analysis and demod engines
- Baseline coding analysis and decoding engine classes

Partial:
- Auto modulation selection confidence and fallback ranking
- Deinterleave/descramble stage standardization
- Protocol-aware payload parsing
- Stage-level telemetry and decode QA metrics

Missing / Not Explicit:
- Generic dechannelization/demultiplexing stage
- Unified output adapter layer for text/mail/audio/video
- Protocol plugin architecture for NOAA/Meteor/Inmarsat/Iridium chains
- Golden-vector protocol validation suite

## 2. Target Objective

Decode RF/satellite signals into understandable artifacts (text, message/mail-like payloads, audio, future video/image pipelines) using a standardized and measurable stage pipeline.

## 3. Execution Roadmap

## Phase A (P0) - Pipeline Output Foundation

Goals:
- Add unified output adapter layer
- Integrate into comprehensive analyzer results
- Add stage-status reporting in analysis output
- Add regression tests

Tasks:
- [x] Implement output adapters module (text/mail/audio artifact extraction)
- [x] Integrate output artifacts into `SignalAnalyzer.analyze_signal_comprehensive`
- [x] Add explicit `stage_status` map to analysis payload
- [x] Add tests for output adapters
- [x] Register new tests in central test runner

## Phase B (P1) - Protocol Plugin Scaffolding

Goals:
- Introduce protocol plugin interfaces
- Bridge existing standalone decoders (example: Inmarsat)

Tasks:
- [x] Add protocol plugin base classes and registry
- [x] Add Inmarsat-C protocol plugin adapter (sync/frame parser based)
- [x] Add protocol parser result schema in analysis payload (`protocol_outputs`)

## Phase C (P1/P2) - Decode Depth Upgrades

Goals:
- Improve practical decode success in low-SNR and burst channels

Tasks:
- [x] Multi-hypothesis modulation/coding selection with score history
- [x] Explicit deinterleave/descramble stage APIs
- [x] Add dechannelization strategy hooks (TDMA/FDMA style)

## Phase D (P2) - Output Expansion and Productization

Goals:
- Extend beyond bitstream-level outputs

Tasks:
- [x] Add PCM export adapter and WAV writer path
- [x] Add image/video-oriented output contracts
- [x] Add session export + replay + decode report generation

## 4. What Was Implemented In This Iteration

1. Added `rf_spectrum_analyzer/dsp/output_adapters.py`
   - bitstream-to-bytes utility
   - text artifact extraction
   - mail-like message extraction
   - audio metadata artifact extraction

2. Updated `rf_spectrum_analyzer/dsp/signal_analysis.py`
   - integrated output artifact extraction in comprehensive analysis
   - added stage-status map in result payload
   - added helper methods for selecting decoded bits/audio sources

3. Added `rf_spectrum_analyzer/tests/test_output_adapters.py`
   - verifies text artifact extraction
   - verifies mail artifact extraction
   - verifies audio artifact extraction

4. Updated `rf_spectrum_analyzer/tests/run_tests.py`
   - registered `output_adapters` module
   - included it in fast test category

5. Added `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - protocol plugin base interface
   - plugin registry with confidence-based selection
   - Inmarsat-C plugin with sync/frame parsing and protocol text artifacts

6. Updated `rf_spectrum_analyzer/dsp/signal_analysis.py`
   - integrated protocol plugin decode stage
   - added `protocol_outputs` into comprehensive analysis payload
   - merged protocol artifacts into `decoded_outputs`

7. Added protocol tests
   - `rf_spectrum_analyzer/tests/test_protocol_plugins.py`
   - `rf_spectrum_analyzer/tests/test_protocol_pipeline_e2e.py` (IQ -> demod -> decode -> artifacts)

8. Updated test runner registration
   - fast suite includes `protocol_plugins`
   - medium suite includes `protocol_pipeline_e2e`

9. Added `rf_spectrum_analyzer/dsp/decode_stages.py`
   - explicit decode depth pipeline API
   - conservative deinterleave and protocol-aware descramble hooks
   - stage telemetry metrics for dashboard integration

10. Updated `rf_spectrum_analyzer/dsp/signal_analysis.py`
   - integrated decode depth stage before protocol parse
   - added `decode_depth` metrics into analysis payload
   - added `decode_quality` metrics (`frame_count`, uncertain ratio, protocol confidence, artifact count)

11. Added `rf_spectrum_analyzer/tests/test_decode_depth_stages.py`
   - decode stage API tests
   - decode quality metrics tests

12. Updated test runner registration
   - fast suite includes `decode_depth_stages`

13. Updated `rf_spectrum_analyzer/dsp/signal_analysis.py`
   - added multi-hypothesis modulation selection with ranked scores
   - added multi-hypothesis coding selection with score history
   - added TDMA/FDMA dechannelization strategy hooks before protocol parsing
   - extended `decode_quality` with BER/PER/CRC/frame-lock extraction when protocol counters are available

14. Added `rf_spectrum_analyzer/tests/test_decode_hypothesis_and_hooks.py`
   - verifies modulation/coding score history
   - verifies dechannelization hook metrics
   - verifies decode quality BER/PER/CRC/frame-lock fields

15. Updated test runner registration
   - fast suite includes `decode_hypothesis_hooks`

16. Updated `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - Inmarsat plugin now emits native internal counters from frame scan/descramble stage
   - exposes `ber`, `per`, `frame_lock_ratio`, `frame_locks`, `frame_total`, `counter_source`
   - now includes a real CRC validation stage for Inmarsat frames using `CRC-16/CCITT-FALSE`
   - exposes native `crc_ok`, `crc_ok_rate`, `crc_expected`, `crc_computed`, and `crc_algorithm`
   - provides reusable base helper for native counter aggregation so future NOAA/Meteor/Iridium plugins can follow the same pattern

17. Updated protocol tests
   - `rf_spectrum_analyzer/tests/test_protocol_plugins.py` validates native protocol counters
   - `rf_spectrum_analyzer/tests/test_protocol_pipeline_e2e.py` validates counter propagation through analyzer path with scrambled on-air frames and CRC-valid payloads

18. Extended `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - added `IridiumBurstProtocolPlugin` as the first next protocol plugin after Inmarsat
   - uses the shared native-counter helper to keep `ber/per/crc_ok_rate/frame_lock_ratio/counter_source` schema-consistent
   - implements baseline burst segmentation and strict burst validation metrics suitable for current roadmap stage

19. Extended `rf_spectrum_analyzer/tests/test_protocol_plugins.py`
   - validates Iridium plugin selection in L-band conditions
   - validates schema-consistent native counters for Iridium baseline plugin
   - validates burst-log artifact emission

20. Extended `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - added `MeteorLrptProtocolPlugin` baseline for digital packet path
   - reuses shared native-counter helper with the same contract (`ber/per/crc_ok_rate/frame_lock_ratio/counter_source`)
   - adds baseline LRPT frame-sync scanning and packet-log artifacts

21. Extended `rf_spectrum_analyzer/tests/test_protocol_plugins.py`
   - validates Meteor plugin selection in VHF LRPT-like conditions
   - validates schema-consistent native counters for Meteor baseline plugin
   - validates LRPT packet-log artifact emission

22. Extended `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - upgraded `MeteorLrptProtocolPlugin` decode chain with de-randomization and FEC placeholders
   - added detailed quality counters (`derand_improvement_avg`, `fec_confidence_avg`, `fec_symbol_fix_rate_avg`)
   - preserved core native-counter contract (`ber/per/crc_ok_rate/frame_lock_ratio/counter_source`)

23. Extended `rf_spectrum_analyzer/dsp/protocol_plugins.py`
   - added `NoaaAptProtocolPlugin` analog/image baseline
   - implements FM-demod surrogate stage, line-sync detection, and grayscale image artifact extraction
   - emits `image` and `audio` artifacts with stage metadata (`fm_demod_stage`, `line_sync_stage`)

24. Extended `rf_spectrum_analyzer/tests/test_protocol_plugins.py`
   - validates NOAA APT plugin selection for FM VHF conditions
   - validates NOAA APT image/audio artifact emission and stage metadata
   - validates upgraded Meteor decode-chain detail fields

25. Extended UI/export path for NOAA image artifacts
   - `rf_spectrum_analyzer/gui/main_window.py` adds File menu action to export latest decoded image artifact
   - `rf_spectrum_analyzer/core/app.py` captures latest `image` artifact from analysis outputs and routes export requests
   - `rf_spectrum_analyzer/utils/file_io.py` adds `export_artifact_image` for png/json/npy export formats

26. Replaced Meteor placeholders with staged real decode chain
   - added `rf_spectrum_analyzer/dsp/meteor_lrpt_chain.py` with real block deinterleave, hard-decision Viterbi decode (rate 1/2, K=7), and Reed-Solomon syndrome decode/correction stage
   - `rf_spectrum_analyzer/dsp/protocol_plugins.py` now uses chain metadata `deinterleave->viterbi->reed_solomon`
   - preserves native core counters while adding stage-quality metrics (`viterbi_metric_avg`, `rs_corrected_symbols_avg`, `rs_decode_success_rate`)

27. Added golden/contract tests for new stages and export flow
   - `rf_spectrum_analyzer/tests/test_meteor_decode_chain.py` verifies stage-level golden fixtures for deinterleave, Viterbi, RS, and full-chain recovery
   - `rf_spectrum_analyzer/tests/test_output_adapters.py` verifies image artifact export via `DataExporter`
   - `rf_spectrum_analyzer/tests/test_app_layer_integration.py` verifies NOAA image artifact propagation to UI and export invocation
   - `rf_spectrum_analyzer/tests/run_tests.py` registers `meteor_decode_chain` in fast suite

28. Upgraded Meteor RS correction depth for multi-symbol robustness
   - `rf_spectrum_analyzer/dsp/meteor_lrpt_chain.py` updates RS path to Berlekamp-Massey locator + Chien search position resolution and robust multi-symbol magnitude solve over GF(256)
   - aligns RS metadata contract with algorithm tag `rs_bm_forney`
   - fixed Chien inverse-power mapping for LRPT frame index orientation
   - added/updated golden tests in `rf_spectrum_analyzer/tests/test_meteor_decode_chain.py` for single-symbol and multi-symbol correction assertions
   - validated with `meteor_decode_chain`, `protocol_plugins`, and full `--fast` suite

29. Completed NOAA live-view + session report productization and decode-depth fixture expansion
   - `rf_spectrum_analyzer/gui/main_window.py` adds NOAA Image Artifacts dock (history + preview), plus File menu export for session decode report
   - `rf_spectrum_analyzer/core/app.py` now pushes decoded image artifacts to live viewer and records compact per-analysis session snapshots
   - `rf_spectrum_analyzer/utils/file_io.py` adds `export_decode_session_report` with trend series (`snr/ber/per/crc/frame_lock/artifact_count`) and artifact references
   - `rf_spectrum_analyzer/dsp/meteor_lrpt_chain.py` adds soft-decision Viterbi metric path and confidence telemetry
   - `rf_spectrum_analyzer/dsp/protocol_plugins.py` now exposes Meteor soft Viterbi aggregate metrics and NOAA sync-tolerant line detection for edge cases
   - tests extended in `test_meteor_decode_chain.py`, `test_protocol_plugins.py`, `test_output_adapters.py`, `test_app_layer_integration.py`
   - validated by targeted modules and full fast suite (`186` tests pass)

30. Completed Phase D (P2) output expansion and productization
   - `rf_spectrum_analyzer/dsp/output_adapters.py` adds PCM artifact adapter (`pcm_s16le`) and unified image/video/audio artifact contract normalization
   - `rf_spectrum_analyzer/utils/file_io.py` adds WAV export path (`export_pcm_wav_from_artifact`) and decode report replay import (`import_decode_session_report`)
   - `rf_spectrum_analyzer/gui/main_window.py` adds File menu actions for WAV export and session report load/replay
   - `rf_spectrum_analyzer/core/app.py` now tracks latest PCM artifact, supports WAV export, and replays loaded session reports into NOAA image viewer history
   - session snapshots now preserve replayable image preview data in artifact references
   - tests expanded in `test_output_adapters.py` and `test_app_layer_integration.py` for PCM/WAV and session replay paths

## 5. Next Immediate Actions (Recommended)

1. Add optional session replay loader that rehydrates previously exported decode reports into the NOAA/Meteor viewer panels.
2. Expand decode session report to include per-plugin histogram blocks (protocol confidence and uncertainty bands).
3. Add medium-suite E2E fixture that verifies UI menu actions can export both latest image and session report in one run.
