"""
Setup script for GDM-Net package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gdmnet",
    version="0.1.0",
    author="GDM-Net Team",
    author_email="gdmnet@example.com",
    description="Graph-Augmented Dual Memory Network for Multi-Document Understanding",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gdmnet/gdmnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "nlp": [
            "spacy>=3.4.0",
            "nltk>=3.7",
        ],
        "graph": [
            "networkx>=2.8",
            "dgl>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gdmnet-train=train.train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gdmnet": ["*.yaml", "*.json"],
    },
    keywords="graph neural networks, document understanding, multi-hop reasoning, knowledge graphs",
    project_urls={
        "Bug Reports": "https://github.com/gdmnet/gdmnet/issues",
        "Source": "https://github.com/gdmnet/gdmnet",
        "Documentation": "https://gdmnet.readthedocs.io/",
    },
)
