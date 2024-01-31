#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
from os.path import join

from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info

    config = Configuration(None, parent_package, top_path)

    config.add_extension('_custom_module',
                         sources=["python_custom_module.c"],
                         include_dirs=get_numpy_include_dirs())

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
