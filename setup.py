"""
AVCS DNA-MATRIX SPIRIT v7.0 - Setup Configuration
Advanced Vibration Control System with AI-Driven Intelligence
"""

from setuptools import setup, find_packages
import pathlib

# Read the contents of README.md
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else "AVCS DNA-MATRIX SPIRIT - Advanced Vibration Control System"

# Read requirements
requirements = []
if (here / "requirements.txt").exists():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="avcs-dna-matrix-spirit",
    version="7.0.0",
    description="AVCS DNA-MATRIX SPIRIT - Advanced Vibration Control System with AI-Driven Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AVCS Engineering Team",
    author_email="engineering@avcs-systems.com",
    url="https://github.com/avcs-systems/avcs-dna-matrix-spirit",
    
    # Project classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Oil & Gas Industry",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
    ],
    
    keywords=[
        "vibration control",
        "industrial ai",
        "predictive maintenance", 
        "digital twin",
        "fpso",
        "oil and gas",
        "industrial automation",
        "machine learning",
        "condition monitoring",
        "active vibration",
        "mr damper",
        "adaptive control"
    ],
    
    # Package discovery - включаем все ваши модули
    packages=find_packages(include=[
        "adaptive_learning", 
        "adaptive_learning.*",
        "digital_twin",
        "digital_twin.*", 
        "industrial_core",
        "industrial_core.*",
        "plc_integration",
        "plc_integration.*",
        "ui",
        "ui.*"
    ]),
    python_requires=">=3.8, <4",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies for specific features
    extras_require={
        "industrial": [
            "opcua>=0.98.13",
            "pymodbus>=3.6.3", 
            "paho-mqtt>=2.1.0",
        ],
        "ml": [
            "scikit-learn>=1.5.2",
            "torch>=2.4.1",
            "transformers>=4.45.2",
        ],
        "voice": [
            "SpeechRecognition>=3.10.4",
            "pyttsx3>=2.90",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "docker": [
            "docker>=6.0.0",
        ],
    },
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "adaptive_learning": [
            "models/*.joblib",
            "config/*.json",
        ],
        "digital_twin": [
            "config/*.json",
            "models/*.pkl",
        ],
        "industrial_core": [
            "config/*.json",
        ],
        "ui": [
            "assets/*.png",
            "assets/*.css",
        ],
        "assets": [
            "*.png",
            "*.ico",
        ],
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "avcs-dashboard=ui.dashboard:main",
            "avcs-app=avcs_dna_matrix_spirit_app:main",
            "avcs-train=adaptive_learning.adaptive_core:train_cli",
            "avcs-simulate=adaptive_learning.sample_data:simulate_cli",
        ],
    },
    
    # Project URLs
    project_urls={
        "Documentation": "https://avcs-systems.github.io/avcs-dna-matrix-spirit/",
        "Source": "https://github.com/avcs-systems/avcs-dna-matrix-spirit",
        "Tracker": "https://github.com/avcs-systems/avcs-dna-matrix-spirit/issues",
    },
    
    # License
    license="Proprietary",
    
    # Additional metadata
    platforms=["Linux", "Windows", "macOS"],
)
