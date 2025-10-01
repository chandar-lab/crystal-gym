import os

import setuptools

with open("version.txt") as f:
    VERSION = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crystal-gym",
    version=VERSION,
    description="A Gymnasium environment for generating crystalline materials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/chandar-lab/crystal-design",
    # project_urls={
    #     "Bug Tracker": "https://github.com/chandar-lab/crystal-design/issues",
    # },
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "pymatgen>=2025.1.0",
        "ase>=3.23.0",
        "dgl>=2.4.0",
        "matgl>=0.9.0",
        "chgnet>=0.3.8",
        "hydra-core>=1.3.0",
        "wandb>=0.17.0",
        "torchrl>=0.8.0",
        "tensorboard>=2.18.0",
        "pyyaml",
        "h5py",
    ],
    packages=['crystal_gym']
)
