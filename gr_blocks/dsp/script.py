# Let me create the comprehensive RF spectrum analyzer project structure
import os

# Create the project directory structure
project_dirs = [
    "rf_spectrum_analyzer",
    "rf_spectrum_analyzer/config",
    "rf_spectrum_analyzer/core",
    "rf_spectrum_analyzer/backends",
    "rf_spectrum_analyzer/gui",
    "rf_spectrum_analyzer/gui/dialogs",
    "rf_spectrum_analyzer/dsp",
    "rf_spectrum_analyzer/utils",
    "rf_spectrum_analyzer/resources",
    "rf_spectrum_analyzer/resources/icons",
    "rf_spectrum_analyzer/resources/ui",
    "rf_spectrum_analyzer/resources/themes"
]

# Create directories
for dir_path in project_dirs:
    os.makedirs(dir_path, exist_ok=True)
    # Create __init__.py files for Python packages
    if not dir_path.endswith(("resources", "icons", "ui", "themes")):
        with open(os.path.join(dir_path, "__init__.py"), "w") as f:
            f.write("# RF Spectrum Analyzer Package\n")

print("Project directory structure created successfully!")
print("Created directories:")
for dir_path in project_dirs:
    print(f"  {dir_path}")