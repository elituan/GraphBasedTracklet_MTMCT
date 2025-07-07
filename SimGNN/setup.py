from setuptools import setup, find_packages

setup(
    name="SimGNN",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
    ],
    description="Graph Neural Network for trajectory similarity",
) 