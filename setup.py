"""Setup for PyTorch Issues Classifier."""

from setuptools import setup, find_packages

setup(
    name="pytorch-issues-classifier",
    version="1.0.0",
    description="Multi-label classification for PyTorch GitHub issues",
    author="Your Name",
    python_requires=">=3.8",
    install_requires=[
        "pyarrow==14.0.1",
        "datasets==2.18.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "requests",
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
