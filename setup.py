from setuptools import setup

setup(
    name='hict_utils',
    version='1.0rc1.dev1',
    packages=['hict_utils', 'hict_utils.cool_to_hict'],
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='Preliminary version of utilities for HiCT interactive Hi-C scaffolding tool.',
    install_requires=[
        'hict~=1.0rc1.dev1',
        'h5py~=3.7.0',
        'scipy~=1.8.1',
        'numpy~=1.23.1',
        'setuptools~=63.2.0',
        'wheel~=0.37.1',
        'argparse~=1.4.0',
    ],
    entry_points={
        'console_scripts': ['hict_utils=hict_utils:main'],
    }
)
