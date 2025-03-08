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
    python_requires=">=3.10",
)
