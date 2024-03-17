# This is a setup file for the package

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ev2gym',
    version='0.0.1',
    description='A realistic V2G simulator environment',
    author='Stavros Orfanoudakis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='gym, Reinforcement Learning, V2X, V2G, G2V, EVs, ev2gym, Electric Vehicles, Electric Vehicle Simulator',
    # package_dir = {"": "ev2gym"},
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        '': ['*.csv','*.json','*.yaml','*.npy','*.png']
    },
    include_package_data=True,
    install_requires=[
        'gymnasium',
        'pyyaml',
        'matplotlib',
        'pandas',
        'networkx',
        'gurobipy',
    ]
)

"""
rm -rf build dist *.egg-info
py -m build
python3 -m twine upload --repository pypi dist/*
username: __token__
use pypy-api-token
"""

