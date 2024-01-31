import numpy as np
import numba
from Geometry import Vector, is_number


class Matrix3x3(object):

    def __init__(self, m11=0, m12=0, m13=0, m21=0, m22=0, m23=0, m31=0, m32=0, m33=0):
        self._matrix = np.ndarray((3, 3))
        self._matrix[0, 0] = m11
        self._matrix[0, 1] = m12
        self._matrix[0, 2] = m13
        self._matrix[1, 0] = m21
        self._matrix[1, 1] = m22
        self._matrix[1, 2] = m23
        self._matrix[2, 0] = m31
        self._matrix[2, 1] = m32
        self._matrix[2, 2] = m33

    @staticmethod
    def create_from_ndarray(array):
        return Matrix3x3(array[0, 0], array[0, 1], array[0, 2],
                         array[1, 0], array[1, 1], array[1, 2],
                         array[2, 0], array[2, 1], array[2, 2],)

    @property
    def T(self):
        """
        Get transposed matrix
        :return: matrix
        """
        return self.create_from_ndarray(np.transpose(self._matrix))

    @numba.jit
    def determinant(self):
        """
        Gets the determinant of the matrix
        :return: determinant
        """
        return np.linalg.det(self._matrix)

    @numba.jit
    def inverse(self):
        """
        Gets the inverse matrix of the matrix given that it is not singular
        :return: inverse matrix
        """
        det = self.determinant()

        if np.abs(det) < 0.00000001:
            raise Exception('Singular matrices cannot be inverted.')

        return self.create_from_ndarray(np.linalg.inv(self._matrix))

    @numba.jit
    def multiply_by_factor(self, factor):
        """
        Multiply matrix by a number
        :param factor: number
        :return: matrix multiplied by a number
        """
        return self.create_from_ndarray(self._matrix * factor)

    @numba.jit
    def vectmul(self, vector):
        """
        Multiply matrix by a vector
        :param vector: vector to multiply matrix with
        :return: vector as result of the multiplication
        """
        return Vector.create_from_ndarray(np.matmul(self._matrix, vector._vector))

    @numba.jit
    def matmul(self, other):
        """
        Multiply matrix with other matrix
        :param other: other matrix
        :return: matrix product
        """
        return self.create_from_ndarray(np.matmul(self._matrix, other._matrix))

    def __getitem__(self, item):
        """
        Get entry of matrix
        :param item: index or index tuple
        :return: entry
        """
        r, c = 0, 0
        if type(item) == tuple:
            r, c = item
        elif type(item) == list:
            r, c = item[0], item[1]
        else:
            r = item // 3
            c = item % 3
        return self._matrix[r, c]

    def __setitem__(self, key, value):
        """
        Set entry of matrix
        :param key: index or index tuple
        :param value: value to set
        """
        r, c = 0, 0
        if type(key) == tuple:
            r, c = key
        elif type(key) == list:
            r, c = key[0], key[1]
        else:
            r = key // 3
            c = key % 3
        self._matrix[r, c] = value

    def __str__(self):
        return '[{0}\t{1}\t{2}]\n[{3}\t{4}\t{5}]\n[{6}\t{7}\t{8}]'.format(self._matrix[0, 0], self._matrix[0, 1], self._matrix[0, 2],
                                                                          self._matrix[1, 0], self._matrix[1, 1], self._matrix[1, 2],
                                                                          self._matrix[2, 0], self._matrix[2, 1], self._matrix[2, 2])

    def __mul__(self, other):
        if is_number(other):
            return self.multiply_by_factor(other)
        elif type(other) == Vector:
            return self.vectmul(other)
        elif type(other) == Matrix3x3:
            return self.matmul(other)
        else:
            raise Exception('Cannot multiply matrix with object of type "{}".'.format(type(other)))

    @numba.jit
    def __add__(self, other):
        return self.create_from_ndarray(self._matrix + other._matrix)

    @numba.jit
    def __iadd__(self, other):
        return self + other

    @numba.jit
    def __sub__(self, other):
        return self.create_from_ndarray(self._matrix - other._matrix)

    @numba.jit
    def __isub__(self, other):
        return self - other

    @numba.jit
    def __imul__(self, other):
        return self * other

    @numba.jit
    def __rmul__(self, other):
        if is_number(other):
            return self * other
        else:
            return other * self


def identity():
    """
    Returns the identity matrix.
    :return: identity matrix
    """
    return Matrix3x3(m11=1, m22=1, m33=1)


def diagonal(vector: Vector):
    """
    Returns a diagonal matrix with the components of a given vector on its diagonal.
    :param vector: vector for diagonal components
    :return: diagonal matrix
    """
    return Matrix3x3(m11=vector.x, m22=vector.y, m33=vector.z)


def rotation_x_axis(angle: float):
    """
    Returns rotation matrix for rotating vector around the x axis
    :param angle: angle (in radians)
    :return: rotation matrix
    """
    return Matrix3x3(1, 0, 0,
                     0, np.cos(angle), -np.sin(angle),
                     0, np.sin(angle), np.cos(angle))


def rotation_y_axis(angle: float):
    """
    Returns rotation matrix for rotating vector around the y axis
    :param angle: angle (in radians)
    :return: rotation matrix
    """
    return Matrix3x3(np.cos(angle), 0, -np.sin(angle),
                     0, 1, 0,
                     np.sin(angle), 0, np.cos(angle))


def rotation_z_axis(angle: float):
    """
    Returns rotation matrix for rotating vector around the z axis
    :param angle: angle (in radians)
    :return: rotation matrix
    """
    return Matrix3x3(np.cos(angle), -np.sin(angle), 0,
                     np.sin(angle), np.cos(angle), 0,
                     0, 0, 1)

