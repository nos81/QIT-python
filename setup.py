#! /usr/bin/python
from setuptools import setup
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

# Read the version number from a source file.
def find_version(*file_paths):
    # Use codecs.open for Python 2 compatibility
    with codecs.open(os.path.join(here, *file_paths), 'r', 'utf_8') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Use README.txt as the long description
with codecs.open('README.txt', 'r', 'utf_8') as f:
    long_description = f.read()    


setup(name             = 'qit',
      version          = find_version('qit', '__init__.py'),
      description      = 'Quantum Information Toolkit',
      long_description = long_description,
      url              = 'http://qit.sourceforge.net/',
      author           = 'Ville Bergholm et al.',
      author_email     = 'smite-meister@users.sourceforge.net',
      classifiers      = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
        ],
      keywords = 'quantum information, quantum mechanics, toolkit',
      packages         = ['qit'],
      #package_data     = {'qit': ['../doc/*.rst', '../doc/conf.py', '../doc/Makefile', '../doc/make.bat']},
      install_requires = ['numpy>=1.7.1', 'scipy>=0.11', 'matplotlib>=1.2']
      )
