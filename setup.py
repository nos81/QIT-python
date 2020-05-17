#! /usr/bin/python
from setuptools import setup
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

# Read the version number from a source file.
def find_version(*file_paths):
    with open(os.path.join(here, *file_paths), mode='r', encoding='utf_8') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Use README.txt as the long description
with open('README.txt', mode='r', encoding='utf_8') as f:
    long_description = f.read()


setup(
    name             = 'qit',
    version          = find_version('qit', '__init__.py'),
    description      = 'Quantum Information Toolkit',
    long_description = long_description,
    url              = 'http://qit.sourceforge.net/',
    author           = 'Ville Bergholm et al.',
    author_email     = 'smite-meister@users.sourceforge.net',
    classifiers      =
    [
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords = 'quantum information, quantum mechanics, toolkit',
    packages         = ['qit'],
    python_requires  = '>=3.5',
    install_requires = ['numpy>=1.18.4', 'scipy>=1.4.1', 'matplotlib>=3.2.1']
)
