from setuptools import setup, find_packages

setup(
    name="SciMatch",
    version="0.1.0",
    packages=find_packages(),
    author="Melanie Altmann",
    author_email="melanie.altmann@studium.uni-hamburg.de",
    description="SciMatch is a tool helping scientists to connect.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DieEchteMelly/SciMatch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)