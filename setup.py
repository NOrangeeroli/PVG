#!/usr/bin/env python3
"""
Setup script for PVG-Legibility project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pvg-legibility",
    version="0.1.0",
    author="PVG Team",
    description="Reimplementing Prover-Verifier Games for Human-Checkable Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pvg-prepare-data=scripts.prepare_gsm8k:main",
            "pvg-run-round=scripts.run_round:main",
            "pvg-eval=scripts.eval_all:main",
            "pvg-attack=scripts.attack_sneaky_only:main",
        ],
    },
)
