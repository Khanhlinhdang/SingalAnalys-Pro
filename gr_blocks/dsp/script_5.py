# Create utils/logger.py - Logging configuration
logger_content = '''"""
Logging Utilities for RF Spectrum Analyzer

Provides centralized logging configuration with file rotation, console output,
and proper formatting for debugging and monitoring application behavior.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\\033[36m',      # Cyan
        'INFO': '\\033[32m',       # Green
        'WARNING': '\\033[33m',    # Yellow
        'ERROR': '\\033[31m',      # Red
        'CRITICAL': '\\033[35m',   # Magenta
        'RESET': '\\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Create colored log message
        colored_level = f"{log_color}{record.levelname:8s}{reset_color}"
        colored_name = f"{log_color}{record.name}{reset_color}"
        
        # Format message
        formatted_message = super().format(record)
        
        return f"{timestamp} [{colored_level}] {colored_name}: {formatted_message}"


class PerformanceLogger:
    """Performance monitoring logger"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.timers = {}
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer"""
        self.timers[name] = datetime.now()
    
    def end_timer(self, name: str, log_level: int = logging.DEBUG) -> Optional[float]:
        """End a performance timer and log the duration"""
        if name in self.timers:
            duration = (datetime.now() - self.timers[name]).total_seconds()
            self.logger.log(log_level, f"Timer '{name}': {duration:.3f}s")
            del self.timers[name]
            return duration
        else:
            self.logger.warning(f"Timer '{name}' not found")
            return None
    
    def log_memory_usage(self) -> None:
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


class SDRLogger:
    """Specialized logger for SDR operations"""
    
    def __init__(self, logger_name: str = "sdr"):
        self.logger = logging.getLogger(logger_name)
        self.sample_count = 0
        self.error_count = 0
        self.last_report = datetime.now()
    
    def log_samples_processed(self, count: int) -> None:
        """Log number of samples processed"""
        self.sample_count += count
        now = datetime.now()
        
        # Report every 10 seconds
        if (now - self.last_report).total_seconds() >= 10:
            self.logger.info(f"Processed {self.sample_count:,} samples")
            self.sample_count = 0
            self.last_report = now
    
    def log_device_info(self, device_info: dict) -> None:
        """Log SDR device information"""
        self.logger.info("SDR Device Information:")
        for key, value in device_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log SDR-specific errors"""
        self.error_count += 1
        self.logger.error(f"SDR Error [{self.error_count}] {context}: {error}")
    
    def log_frequency_change(self, old_freq: float, new_freq: float) -> None:
        """Log frequency changes"""
        self.logger.info(f"Frequency changed: {old_freq/1e6:.3f} MHz -> {new_freq/1e6:.3f} MHz")
    
    def log_gain_change(self, old_gain: float, new_gain: float) -> None:
        """Log gain changes"""
        self.logger.info(f"Gain changed: {old_gain:.1f} dB -> {new_gain:.1f} dB")


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_performance: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup application logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Enable console output
        enable_performance: Enable performance logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
    """
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter()
    
    # Setup file logging
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    # Setup console logging
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        root_logger.addHandler(console_handler)
    
    # Setup performance logging
    if enable_performance:
        perf_logger = logging.getLogger("performance")
        perf_logger.setLevel(logging.DEBUG)
        
        if log_file:
            perf_log_file = str(Path(log_file).with_suffix('.performance.log'))
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=max_file_size // 2,
                backupCount=backup_count,
                encoding='utf-8'
            )
            perf_handler.setFormatter(detailed_formatter)
            perf_logger.addHandler(perf_handler)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
    
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("pyqtgraph").setLevel(logging.WARNING)


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    return PerformanceLogger()


def get_sdr_logger() -> SDRLogger:
    """Get SDR logger instance"""
    return SDRLogger()


class LogContext:
    """Context manager for scoped logging"""
    
    def __init__(self, logger_name: str, message: str, level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.message = message
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(f"Failed: {self.message} ({duration:.3f}s) - {exc_val}")
        else:
            self.logger.log(self.level, f"Completed: {self.message} ({duration:.3f}s)")


# Example usage
if __name__ == "__main__":
    # Test logging setup
    setup_logging(
        level=logging.DEBUG,
        log_file="test_logs/rf_spectrum.log",
        enable_console=True,
        enable_performance=True
    )
    
    # Test different loggers
    main_logger = logging.getLogger("main")
    main_logger.debug("Debug message")
    main_logger.info("Info message")
    main_logger.warning("Warning message")
    main_logger.error("Error message")
    
    # Test performance logger
    perf_logger = get_performance_logger()
    perf_logger.start_timer("test_operation")
    import time
    time.sleep(0.1)
    perf_logger.end_timer("test_operation")
    
    # Test SDR logger
    sdr_logger = get_sdr_logger()
    sdr_logger.log_device_info({"device": "RTL-SDR", "frequency": "433.92 MHz"})
    sdr_logger.log_samples_processed(1000)
    
    # Test log context
    with LogContext("test", "Test operation"):
        time.sleep(0.05)
    
    print("Logging test completed")
'''

with open("rf_spectrum_analyzer/utils/logger.py", "w") as f:
    f.write(logger_content)

print("Created utils/logger.py")