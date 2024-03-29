from distutils.core import setup,Extension
import numpy.distutils.misc_util

setup(
    ext_modules = [Extension("_smaract", ["python_smaract.c"],
                             include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
                             + ['"C:\Program Files\SmarAct\MCS\SDK\include"'],
                             library_dirs = ['"C:\Program Files\SmarAct\MCS\SDK\lib64"'],
                             libraries = ["MCSControl"],
                             )]
    )