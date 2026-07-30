from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
REQUIREMENTS = [
    line
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line and not line.startswith("#")
]

setup(
    name="utae-water-segmentation",
    version="0.1.0",
    author="Bilel Khlaifi",
    description="Temporal Sentinel-1 and Sentinel-2 water segmentation with PyTorch",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/khlaifiabilel/utae-water-segmentation",
    packages=find_packages(),
    py_modules=["inference_s2", "train"],
    license="GPL-3.0-only",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=REQUIREMENTS,
)
