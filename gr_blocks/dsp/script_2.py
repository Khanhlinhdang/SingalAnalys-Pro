# Create setup.py for project installation
setup_content = '''#!/usr/bin/env python3
"""
RF Spectrum Analyzer - Advanced SDR Signal Processing Application
Integrates pyspectrum, mhostetter/sdr, and scikit-dsp-comm libraries
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt"""
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    return []

def read_long_description():
    """Read long description from README"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Advanced RF Spectrum Analyzer with integrated SDR processing"

setup(
    name="rf-spectrum-analyzer",
    version="1.0.0",
    description="Advanced RF signal acquisition, analysis, and processing software",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="RF Spectrum Team",
    author_email="contact@rfspectrum.dev",
    url="https://github.com/rfspectrum/rf-spectrum-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Communications :: Ham Radio",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=6.0", "pytest-qt>=4.0", "black>=22.0", "flake8>=4.0"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "rf-spectrum-analyzer=rf_spectrum_analyzer.main:main",
            "rfa=rf_spectrum_analyzer.main:main",
        ],
    },
    package_data={
        "rf_spectrum_analyzer": [
            "resources/icons/*.png",
            "resources/ui/*.ui",
            "resources/themes/*.qss",
            "config/*.yaml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
'''

with open("rf_spectrum_analyzer/setup.py", "w") as f:
    f.write(setup_content)

print("Created setup.py")