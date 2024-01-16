# This is a setup file for the package

from setuptools import setup

setup(
    name='EVsSimulator',
    version='0.0.1',
    description='A realistic V2X environment using gym',    
    author='Stavros Orfanoudakis',
    py_modules=['EVsSimulator'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
                      
