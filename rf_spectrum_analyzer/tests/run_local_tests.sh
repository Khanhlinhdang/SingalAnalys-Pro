#!/bin/bash
# Test automation script for local development
# Usage: ./run_local_tests.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_FAST=false
RUN_ALL=false
RUN_COVERAGE=false
RUN_LINT=false
RUN_PERFORMANCE=false
VERBOSE=false
CLEAN=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -f, --fast          Run only fast tests
    -a, --all           Run all tests
    -c, --coverage      Run with coverage analysis
    -l, --lint          Run linting and code quality checks
    -p, --performance   Run performance benchmarks
    -v, --verbose       Verbose output
    --clean             Clean previous test results
    -h, --help          Show this help message

Examples:
    $0 --fast           # Run fast tests only
    $0 --all --coverage # Run all tests with coverage
    $0 --lint           # Run code quality checks
    $0 --clean --all    # Clean and run all tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--fast)
            RUN_FAST=true
            shift
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -c|--coverage)
            RUN_COVERAGE=true
            shift
            ;;
        -l|--lint)
            RUN_LINT=true
            shift
            ;;
        -p|--performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default behavior if no specific options
if [ "$RUN_FAST" = false ] && [ "$RUN_ALL" = false ] && [ "$RUN_LINT" = false ] && [ "$RUN_PERFORMANCE" = false ]; then
    RUN_FAST=true
fi

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

print_status "RF Spectrum Analyzer Test Suite"
print_status "================================"

# Clean previous results if requested
if [ "$CLEAN" = true ]; then
    print_status "Cleaning previous test results..."
    rm -rf test_results/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_success "Cleaned previous test results"
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Not running in a virtual environment"
    print_warning "Consider activating a virtual environment first"
fi

# Install/upgrade required packages
print_status "Checking test dependencies..."
python3 -m pip install --quiet --upgrade pip

# Check if requirements.txt exists
if [ -f "../requirements.txt" ]; then
    python3 -m pip install --quiet -r ../requirements.txt
fi

# Install test-specific requirements
python3 -m pip install --quiet coverage pytest pytest-xvfb xmlrunner memory-profiler psutil matplotlib

# Create test results directory
mkdir -p test_results

# Function to run tests with error handling
run_test_module() {
    local module=$1
    local description=$2
    
    print_status "Running $description..."
    
    if [ "$VERBOSE" = true ]; then
        python3 run_tests.py --module "$module" --verbosity 2
    else
        python3 run_tests.py --module "$module" --ci
    fi
    
    if [ $? -eq 0 ]; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Start testing
TOTAL_ERRORS=0

# Run linting if requested
if [ "$RUN_LINT" = true ]; then
    print_status "Running code quality checks..."
    
    # Check if linting tools are available
    LINT_TOOLS=("flake8" "black" "isort")
    MISSING_TOOLS=()
    
    for tool in "${LINT_TOOLS[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            MISSING_TOOLS+=("$tool")
        fi
    done
    
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        print_warning "Installing missing linting tools: ${MISSING_TOOLS[*]}"
        python3 -m pip install --quiet "${MISSING_TOOLS[@]}"
    fi
    
    # Run flake8
    print_status "Running flake8..."
    if flake8 . --count --statistics --exit-zero; then
        print_success "flake8 check completed"
    else
        print_warning "flake8 found issues"
        ((TOTAL_ERRORS++))
    fi
    
    # Run black check
    print_status "Running black format check..."
    if black --check --diff . &> /dev/null; then
        print_success "black format check passed"
    else
        print_warning "black format check found issues"
        if [ "$VERBOSE" = true ]; then
            black --check --diff .
        fi
        ((TOTAL_ERRORS++))
    fi
    
    # Run isort check
    print_status "Running isort import check..."
    if isort --check-only --diff . &> /dev/null; then
        print_success "isort import check passed"
    else
        print_warning "isort import check found issues"
        if [ "$VERBOSE" = true ]; then
            isort --check-only --diff .
        fi
        ((TOTAL_ERRORS++))
    fi
fi

# Run tests
if [ "$RUN_FAST" = true ]; then
    print_status "Running fast tests..."
    
    if [ "$COVERAGE" = true ]; then
        python3 run_tests.py --fast --coverage
    else
        python3 run_tests.py --fast
    fi
    
    if [ $? -ne 0 ]; then
        ((TOTAL_ERRORS++))
    fi
fi

if [ "$RUN_ALL" = true ]; then
    print_status "Running all tests..."
    
    # Run each test module
    MODULES=("imports" "dsp_filters" "dsp_modulation" "dsp_analysis" "dsp_utils" "core" "backends" "gui" "integration")
    
    for module in "${MODULES[@]}"; do
        if ! run_test_module "$module" "$module tests"; then
            ((TOTAL_ERRORS++))
        fi
    done
    
    # Run with coverage if requested
    if [ "$RUN_COVERAGE" = true ]; then
        print_status "Running coverage analysis..."
        python3 run_tests.py --all --coverage
        
        if [ $? -eq 0 ]; then
            print_success "Coverage analysis completed"
            if [ -f "test_results/coverage_html/index.html" ]; then
                print_status "Coverage report available at: test_results/coverage_html/index.html"
            fi
        else
            print_error "Coverage analysis failed"
            ((TOTAL_ERRORS++))
        fi
    fi
fi

# Run performance tests if requested
if [ "$RUN_PERFORMANCE" = true ]; then
    print_status "Running performance benchmarks..."
    
    if ! run_test_module "debug_performance" "performance benchmarks"; then
        ((TOTAL_ERRORS++))
    fi
fi

# Final summary
print_status "Test Summary"
print_status "============"

if [ $TOTAL_ERRORS -eq 0 ]; then
    print_success "All tests completed successfully!"
    
    # Show test results if available
    if [ -f "test_results/test_report.json" ]; then
        print_status "Detailed test report: test_results/test_report.json"
    fi
    
    exit 0
else
    print_error "Tests completed with $TOTAL_ERRORS error(s)"
    exit 1
fi