#! /usr/bin/python

from distutils.core import setup
from qit import version


with open('README.txt') as file:
    long_description = file.read()    
# TODO cat changelog

setup(name             = 'qit',
      version          = version(),
      author           = 'Ville Bergholm et al.',
      author_email     = 'smite-meister@users.sourceforge.net',
      url              = 'http://qit.sourceforge.net/',
      description      = 'Quantum Information Toolkit',
      long_description = 'QIT is a comprehensive, easy-to-use interactive numerical toolkit for quantum information and computing, available for both MATLAB and Python.',
      classifiers      = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
        ],
      packages         = ['qit'],
      package_data     = {'qit': ['../doc/*.rst', '../doc/conf.py', '../doc/Makefile', '../doc/make.bat']}
      )
