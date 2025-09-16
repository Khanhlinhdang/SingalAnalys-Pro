#!/usr/bin/env python3
"""
Setup script for RF Spectrum Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

setup(
    name="rf-spectrum-analyzer",
    version="1.0.0",
    author="RF Signal Processing Team",
    author_email="contact@rfspectrumanalyzer.com",
    description="Advanced RF Spectrum Analyzer with integrated DSP capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rfteam/rf-spectrum-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications :: Ham Radio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-qt>=4.2.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "usrp": [
            "uhd>=3.15.0",
        ],
        "hackrf": [
            "hackrf>=2018.1.1",
        ],
        "pluto": [
            "plutosdr>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rf-spectrum-analyzer=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rf_spectrum_analyzer": [
            "resources/icons/*",
            "resources/ui/*",
            "resources/themes/*",
        ],
    },
    keywords=[
        "rf", "spectrum", "analyzer", "sdr", "signal processing", 
        "dsp", "radio", "communications", "pyqt", "pyqtgraph"
    ],
    project_urls={
        "Bug Reports": "https://github.com/rfteam/rf-spectrum-analyzer/issues",
        "Source": "https://github.com/rfteam/rf-spectrum-analyzer",
        "Documentation": "https://rf-spectrum-analyzer.readthedocs.io/",
    },
)