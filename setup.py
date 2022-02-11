import io
import runpy
import os
from setuptools import setup, find_packages

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(
    os.path.join(root, 'version.py'))['version']

setup_requires = [
    "setuptools>=18.0",
    ]

install_requires = [
    "h5py==2.8.0",
    "Pillow==5.1.0",
    "terminaltables==3.1.0",
    "redis==2.10.5",
    "numpy>=1.16.0",
    "matplotlib>=3.0.0",
    "scipy>=1.1.0",
    "nengo>=2.8.0",
    "nengo_extras>=0.3.0"
    ]

setup(
    name='masters_thesis',
    version=version,
    description='Pawel Jaworski masters thesis for LLP MPC',
    url='https://github.com/p3jawors/masters_thesis',
    author='Pawel Jaworski',
    author_email='p3jawors@gmail.com',
    license="Free for non-commercial use",
    long_description=read('README.rst'),
    install_requires=install_requires + setup_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
)
