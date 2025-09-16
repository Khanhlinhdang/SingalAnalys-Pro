@echo off
REM Test automation script for Windows local development
REM Usage: run_local_tests.bat [options]

setlocal enabledelayedexpansion

REM Default options
set RUN_FAST=false
set RUN_ALL=false
set RUN_COVERAGE=false
set RUN_LINT=false
set RUN_PERFORMANCE=false
set VERBOSE=false
set CLEAN=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-f" set RUN_FAST=true
if "%~1"=="--fast" set RUN_FAST=true
if "%~1"=="-a" set RUN_ALL=true
if "%~1"=="--all" set RUN_ALL=true
if "%~1"=="-c" set RUN_COVERAGE=true
if "%~1"=="--coverage" set RUN_COVERAGE=true
if "%~1"=="-l" set RUN_LINT=true
if "%~1"=="--lint" set RUN_LINT=true
if "%~1"=="-p" set RUN_PERFORMANCE=true
if "%~1"=="--performance" set RUN_PERFORMANCE=true
if "%~1"=="-v" set VERBOSE=true
if "%~1"=="--verbose" set VERBOSE=true
if "%~1"=="--clean" set CLEAN=true
if "%~1"=="-h" goto :show_usage
if "%~1"=="--help" goto :show_usage
shift
goto :parse_args

:args_done
REM Set default behavior if no specific options
if "%RUN_FAST%"=="false" if "%RUN_ALL%"=="false" if "%RUN_LINT%"=="false" if "%RUN_PERFORMANCE%"=="false" (
    set RUN_FAST=true
)

REM Change to script directory
cd /d "%~dp0"

echo RF Spectrum Analyzer Test Suite
echo ================================

REM Clean previous results if requested
if "%CLEAN%"=="true" (
    echo [INFO] Cleaning previous test results...
    if exist test_results rmdir /s /q test_results
    if exist .pytest_cache rmdir /s /q .pytest_cache
    if exist __pycache__ rmdir /s /q __pycache__
    for /r %%i in (*.pyc) do del "%%i" 2>nul
    for /f "delims=" %%i in ('dir /s /b /ad __pycache__ 2^>nul') do rmdir /s /q "%%i" 2>nul
    echo [SUCCESS] Cleaned previous test results
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Not running in a virtual environment
    echo [WARNING] Consider activating a virtual environment first
)

REM Install/upgrade required packages
echo [INFO] Checking test dependencies...
python -m pip install --quiet --upgrade pip

REM Check if requirements.txt exists
if exist "..\requirements.txt" (
    python -m pip install --quiet -r ..\requirements.txt
)

REM Install test-specific requirements
python -m pip install --quiet coverage pytest xmlrunner memory-profiler psutil matplotlib

REM Create test results directory
if not exist test_results mkdir test_results

set TOTAL_ERRORS=0

REM Run linting if requested
if "%RUN_LINT%"=="true" (
    echo [INFO] Running code quality checks...
    
    REM Install linting tools if needed
    python -m pip install --quiet flake8 black isort
    
    REM Run flake8
    echo [INFO] Running flake8...
    flake8 . --count --statistics --exit-zero
    if errorlevel 1 (
        echo [WARNING] flake8 found issues
        set /a TOTAL_ERRORS+=1
    ) else (
        echo [SUCCESS] flake8 check completed
    )
    
    REM Run black check
    echo [INFO] Running black format check...
    black --check --diff . >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] black format check found issues
        if "%VERBOSE%"=="true" black --check --diff .
        set /a TOTAL_ERRORS+=1
    ) else (
        echo [SUCCESS] black format check passed
    )
    
    REM Run isort check
    echo [INFO] Running isort import check...
    isort --check-only --diff . >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] isort import check found issues
        if "%VERBOSE%"=="true" isort --check-only --diff .
        set /a TOTAL_ERRORS+=1
    ) else (
        echo [SUCCESS] isort import check passed
    )
)

REM Run tests
if "%RUN_FAST%"=="true" (
    echo [INFO] Running fast tests...
    
    if "%RUN_COVERAGE%"=="true" (
        python run_tests.py --fast --coverage
    ) else (
        python run_tests.py --fast
    )
    
    if errorlevel 1 set /a TOTAL_ERRORS+=1
)

if "%RUN_ALL%"=="true" (
    echo [INFO] Running all tests...
    
    REM Run each test module
    set modules=imports dsp_filters dsp_modulation dsp_analysis dsp_utils core backends gui integration
    
    for %%m in (%modules%) do (
        echo [INFO] Running %%m tests...
        
        if "%VERBOSE%"=="true" (
            python run_tests.py --module %%m --verbosity 2
        ) else (
            python run_tests.py --module %%m --ci
        )
        
        if errorlevel 1 (
            echo [ERROR] %%m tests failed
            set /a TOTAL_ERRORS+=1
        ) else (
            echo [SUCCESS] %%m tests completed successfully
        )
    )
    
    REM Run with coverage if requested
    if "%RUN_COVERAGE%"=="true" (
        echo [INFO] Running coverage analysis...
        python run_tests.py --all --coverage
        
        if errorlevel 1 (
            echo [ERROR] Coverage analysis failed
            set /a TOTAL_ERRORS+=1
        ) else (
            echo [SUCCESS] Coverage analysis completed
            if exist "test_results\coverage_html\index.html" (
                echo [INFO] Coverage report available at: test_results\coverage_html\index.html
            )
        )
    )
)

REM Run performance tests if requested
if "%RUN_PERFORMANCE%"=="true" (
    echo [INFO] Running performance benchmarks...
    
    if "%VERBOSE%"=="true" (
        python run_tests.py --module debug_performance --verbosity 2
    ) else (
        python run_tests.py --module debug_performance --ci
    )
    
    if errorlevel 1 (
        echo [ERROR] Performance benchmarks failed
        set /a TOTAL_ERRORS+=1
    ) else (
        echo [SUCCESS] Performance benchmarks completed successfully
    )
)

REM Final summary
echo [INFO] Test Summary
echo [INFO] ============

if %TOTAL_ERRORS% equ 0 (
    echo [SUCCESS] All tests completed successfully!
    
    REM Show test results if available
    if exist "test_results\test_report.json" (
        echo [INFO] Detailed test report: test_results\test_report.json
    )
    
    exit /b 0
) else (
    echo [ERROR] Tests completed with %TOTAL_ERRORS% error(s)
    exit /b 1
)

:show_usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo     -f, --fast          Run only fast tests
echo     -a, --all           Run all tests
echo     -c, --coverage      Run with coverage analysis
echo     -l, --lint          Run linting and code quality checks
echo     -p, --performance   Run performance benchmarks
echo     -v, --verbose       Verbose output
echo     --clean             Clean previous test results
echo     -h, --help          Show this help message
echo.
echo Examples:
echo     %~nx0 --fast           # Run fast tests only
echo     %~nx0 --all --coverage # Run all tests with coverage
echo     %~nx0 --lint           # Run code quality checks
echo     %~nx0 --clean --all    # Clean and run all tests
echo.
exit /b 0