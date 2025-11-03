from setuptools import setup, find_packages

setup(
    name="avcs_dna_matrix_spirit",
    version="6.0.0",
    author="AVCS Engineering Labs",
    author_email="contact@avcs-labs.io",
    description="AVCS DNA-MATRIX SPIRIT — Industrial Digital Twin & IIoT AI Framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yeruslan72/AVCS-DNA-MATRIX-SPIRIT-v6.0",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "avcs-spirit=avcs_dna_matrix_spirit_app:main",
        ],
    },
)
