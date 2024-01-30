from distutils.core import setup,Extension
import numpy.distutils.misc_util

setup(
    ext_modules = [Extension("_gage", ["CsSdkMisc.c","python_gage.c"],
                             include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
                             + ['"C:\Program Files (x86)\Gage\CompuScope\CompuScope C SDK\C Common"',
                                '"C:\Program Files (x86)\Gage\CompuScope\include"'],
                             library_dirs = ['"C:\Program Files (x86)\Gage\CompuScope\lib64"'],
                             libraries = ["CsAppSupport","CsSsm"],
                             )]
    )