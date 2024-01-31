from Geometry.tools import is_number
from Geometry.native.Vector import Vector, get_containing_orthonormal_basis, get_rotation, cross
from Geometry.native.Matrix import Matrix3x3, rotation_x_axis, rotation_y_axis, rotation_z_axis, identity
from Geometry.Ray import Ray
from Geometry.Sphere import Sphere


BACKENDS = ['native', 'numpy', 'numba']


def set_backend(backend):
    if backend not in BACKENDS:
        raise Exception('The backend "{}" does not exist. Valid choices for backends are: {}'.format(backend, BACKENDS))

    if backend == 'native':
        from Geometry.native.Vector import Vector, get_containing_orthonormal_basis, get_rotation, cross
        from Geometry.native.Matrix import Matrix3x3, rotation_x_axis, rotation_y_axis, rotation_z_axis, identity
    elif backend == 'numpy':
        from Geometry.numpy.Vector import Vector, get_containing_orthonormal_basis, get_rotation, cross
        from Geometry.numpy.Matrix import Matrix3x3, rotation_x_axis, rotation_y_axis, rotation_z_axis, identity
    elif backend == 'numba':
        from Geometry.numba.Vector import Vector, get_containing_orthonormal_basis, get_rotation, cross
        from Geometry.numba.Matrix import Matrix3x3, rotation_x_axis, rotation_y_axis, rotation_z_axis, identity


set_backend('native')
