from setuptools import setup, find_packages

setup(
    name="prompt-zen",
    version="1.0.0",
    description="A framework for iterative prompt refinement and optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Philipp Heller",
    license="BSD-3-Clause",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.3",
        "langchain>=0.3.7",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
