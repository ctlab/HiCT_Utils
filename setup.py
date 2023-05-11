from typing import List
from setuptools import find_packages, setup


requirements: List[str] = []
with open("requirements.txt", mode="rt", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name='hict_utils',
    version='0.1.1rc1',
    packages=list(set(['hict_utils', 'hict_utils.cool_to_hict']).union(find_packages())),
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='Preliminary version of utilities for HiCT interactive Hi-C scaffolding tool.',
    install_requires=list(set([
        'hict>=0.1.1rc1,<1.0',
    ]).union(requirements)),
    entry_points={
        'console_scripts': ['hict_utils=hict_utils:main'],
    }
)
