"""
Logging configuration    def format(self, record):
        # Simple formatting to avoid recursion
        try:
            # Add colors to levelname
            if record.levelname in self.COLORS:
                levelname_with_color = (
                    f"{self.COLORS[record.levelname]}"
                    f"{record.levelname}"
                    f"{self.COLORS['RESET']}"
                )
                # Create new record to avoid modifying original
                import logging
                new_record = logging.LogRecord(
                    record.name, record.levelno, record.pathname, record.lineno,
                    record.msg, record.args, record.exc_info, record.funcName
                )
                new_record.levelname = levelname_with_color
                return super().format(new_record)
            else:
                return super().format(record)
        except:
            # Fallback to simple format
            return f"{record.levelname}: {record.getMessage()}"ectrum Analyzer
Provides centralized logging setup with configurable levels and output formats
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to the log level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)

class Logger:
    """Centralized logging configuration for the application"""
    
    def __init__(self):
        self.loggers = {}
        self.log_dir = Path.home() / '.rf_spectrum_analyzer' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logger(
        self,
        name: str,
        level: str = 'INFO',
        console_output: bool = True,
        file_output: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Setup and configure a logger
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console output
            file_output: Enable file output
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_output:
            log_file = self.log_dir / f'{name}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get an existing logger or create a new one with default settings"""
        if name not in self.loggers:
            return self.setup_logger(name)
        return self.loggers[name]
    
    def set_level(self, name: str, level: str):
        """Set logging level for a specific logger"""
        if name in self.loggers:
            self.loggers[name].setLevel(getattr(logging, level.upper()))
    
    def set_global_level(self, level: str):
        """Set logging level for all configured loggers"""
        log_level = getattr(logging, level.upper())
        for logger in self.loggers.values():
            logger.setLevel(log_level)
    
    def cleanup_old_logs(self, days: int = 30):
        """Remove log files older than specified days"""
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
            
            for log_file in self.log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"Removed old log file: {log_file}")
                    
        except Exception as e:
            print(f"Error cleaning up old logs: {e}")

# Global logger instance
_logger_instance = None

def get_logger(name: str = 'rf_spectrum_analyzer') -> logging.Logger:
    """Get a logger instance with the specified name"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance.get_logger(name)

def setup_application_logging(
    level: str = 'INFO',
    console: bool = True,
    file: bool = True
):
    """Setup logging for the entire application"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    
    # Setup main application logger
    main_logger = _logger_instance.setup_logger(
        'rf_spectrum_analyzer',
        level=level,
        console_output=console,
        file_output=file
    )
    
    # Setup component loggers
    component_loggers = [
        'sdr_backend',
        'signal_processor',
        'gui',
        'spectrum_widget',
        'waterfall_widget',
        'controls_widget'
    ]
    
    for component in component_loggers:
        _logger_instance.setup_logger(
            component,
            level=level,
            console_output=console,
            file_output=file
        )
    
    # Cleanup old logs
    _logger_instance.cleanup_old_logs()
    
    main_logger.info("Application logging initialized")
    return main_logger

def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """Log an exception with full traceback"""
    import traceback
    
    error_msg = f"Exception in {context}: {str(exc)}" if context else f"Exception: {str(exc)}"
    logger.error(error_msg)
    logger.debug("Full traceback:", exc_info=True)

# Decorator for logging function calls
def log_calls(logger_name: str = None):
    """Decorator to log function calls and execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = get_logger(logger_name or func.__module__)
            start_time = datetime.now()
            
            log.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                log.debug(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                log.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = get_logger(logger_name)
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        self.timers[name] = datetime.now()
        self.logger.debug(f"Started timer: {name}")
    
    def stop_timer(self, name: str) -> float:
        """Stop a performance timer and return elapsed time"""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = (datetime.now() - self.timers[name]).total_seconds()
        del self.timers[name]
        
        self.logger.debug(f"Timer '{name}': {elapsed:.3f}s")
        return elapsed
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def log_cpu_usage(self):
        """Log current CPU usage"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logger.debug(f"CPU usage: {cpu_percent:.1f}%")
        except ImportError:
            self.logger.debug("psutil not available for CPU monitoring")

# Global performance monitor
performance_monitor = PerformanceMonitor()