import setuptools

setup_requires = [
    "setuptools>=18.0",
    ]

install_requires = [
    "Pillow==5.1.0",
    "terminaltables==3.1.0",
    "redis==2.10.5",
    "numpy>=1.16.0",
    "matplotlib>=3.0.0",
    "scipy>=1.1.0",
    "nengo>=2.8.0",
    "nengo_extras>=0.3.0"
    "h5py==2.8.0",
    "nni>=2.6.0"
    ]

setuptools.setup(
    name='masters_thesis',
    version='0.0.1',
    description='Pawel Jaworski masters thesis for LLP MPC',
    url='https://github.com/p3jawors/masters_thesis',
    author='Pawel Jaworski',
    author_email='p3jawors@gmail.com',
    license="Free for non-commercial use",
    install_requires=install_requires + setup_requires,
    setup_requires=setup_requires,
    packages=setuptools.find_packages(),
)
