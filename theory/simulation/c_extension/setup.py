#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
from os.path import join

from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info

    config = Configuration(None, parent_package, top_path)

    # Get a local copy of lapack_opt_info
    lapack_opt = dict(get_info('lapack_opt',notfound_action=2))
    # Pop off the libraries list so it can be combined with
    # additional required libraries
    lapack_libs = lapack_opt.pop('libraries', [])

    mach_src = [join('mach','*.f')]
    quadpack_src = [join('quadpack','*.f')]

    config.add_library('mach', sources=mach_src,
                       config_fc={'noopt':(__file__,1)})
    config.add_library('quadpack', sources=quadpack_src)


    # Extensions
    # quadpack:
    include_dirs = [join(os.path.dirname(__file__), '..', '_lib', 'src')]
    if 'include_dirs' in lapack_opt:
        lapack_opt = dict(lapack_opt)
        include_dirs.extend(lapack_opt.pop('include_dirs'))

    config.add_extension('_custom_module',
                         sources=["python_custom_module.c"],
                         libraries=(['quadpack', 'mach'] + lapack_libs),
                         depends=quadpack_src + mach_src,
                         include_dirs=get_numpy_include_dirs(),
                         **lapack_opt)


    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
