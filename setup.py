from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="utae-water-segmentation",
    version="0.1.0",
    author="Bilel khlaifia",
    author_email="khlaifiabilel@icloud.com",
    description="UTAE-PAPS model for water/land segmentation using Sentinel-1 and Sentinel-2 data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khlaifiabilel/utae-water-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
        "transformers>=4.20.0",
        "rasterio>=1.3.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
)