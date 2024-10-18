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
        "torch",
    ],
    packages=['crystal_gym']
)
