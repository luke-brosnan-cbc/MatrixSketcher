# setup.py

from setuptools import setup, find_packages

setup(
    name="matrixsketcher",
    version="0.1.1",
    description="A collection of efficient matrix sketching methods",
    author="Luke Brosnan",
    author_email="luke.brosnan.cbc@gmail.com",
    url="https://github.com/luke-brosnan-cbc/MatrixSketcher",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
        classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
)
