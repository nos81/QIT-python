#! /usr/bin/python

from distutils.core import setup
from qit import version


setup(name          = 'qit',
      version       = version(),
      author        = 'Ville Bergholm',
      author_email  = 'smite-meister@users.sourceforge.net',
      url           = 'http://qit.sourceforge.net/',
      description   = 'Quantum Information Toolkit is a comprehensive, easy-to-use interactive numerical toolkit for quantum information and computing, available for both MATLAB and Python.',
      classifiers   = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
        ],
      packages      = ['qit'],
      provides      = ['qit']
      )

# TODO package_data (docs!)
