from setuptools import setup, find_packages

setup(
    name="jssp_sim",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "jssp_sim=jssp_sim.main:main",
        ],
    },
    python_requires=">=3.9",
    author="JSSP-Sim Team",
    author_email="example@example.com",
    description="A Job Shop Scheduling Problem Simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/jssp_sim",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
) 