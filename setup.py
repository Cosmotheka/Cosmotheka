#!/usr/bin/python3

from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

with open("requirements.txt", 'r') as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="xCell",
    version="dev",
    description=("Pipeline to compute Nx2pt angular power spectra and their covariances."),
    license="GPLv2",
    keywords="angular power spectra covariances Nx2pt",
    url="https://github.com/xC-ell/xCell",
    packages=['xcell'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Physics"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    install_requires=requirements,
    python_requires='>3',
)
