import numpy as np
from math import isnan


def _is_number_(candidate):
    """
    Returns true if the given value is a number
    :param candidate: value
    :return: true or false
    """
    compare_type = type(candidate)
    return compare_type == int or compare_type == float or compare_type == np.int or compare_type == np.float or compare_type == np.float16 or compare_type == np.float32 or compare_type == np.float64


def cross(a, b):
    """
    Returns the cross product of two vectors
    :param a: first vector
    :param b: second vector
    :return: cross product
    """
    return Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


class Vector(object):
    """
    A vector with 3 components
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def clone(self):
        """
        Clones the vector. Useful to prevent accidental manipulation.
        :return: Clone of vector
        """
        return Vector(self.x, self.y, self.z)

    def magnitude_sqrt(self):
        """
        Returns the squared magnitude of the vector
        :return: number representing the squared magnitude
        """
        return self.x * self.x + self.y * self.y + self.z * self.z

    def magnitude(self):
        """
        Returns the magnitude of the vector
        :return: Number representing the magnitude
        """
        return np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        """
        Normalize the vectors magnitude to 1
        """
        magnitude = self.magnitude()
        self.x /= magnitude
        self.y /= magnitude
        self.z /= magnitude

    def normalized(self):
        """
        Return normalized the vector
        """
        result = self.clone()
        result.normalize()
        return result

    def is_nan(self):
        """
        Returns true if one or multiple components of the vector are not a number
        :return: true or false
        """
        return isnan(self.x) or isnan(self.y) or isnan(self.z)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise IndexError('There are only 3 components in this vector.')

    def __str__(self):
        return '[{}]\n|{}|\n[{}]'.format(self.x, self.y, self.z)

    def __iadd__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __isub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __mul__(self, other):
        if _is_number_(other):
            return Vector(self.x * other, self.y * other, self.z * other)
        elif type(other) == Vector:
            return self @ other

    def __rmul__(self, other):
        if _is_number_(other):
            return self * other

    def __truediv__(self, other):
        if _is_number_(other):
            factor = 1 / other
            return Vector(self.x * factor, self.y * factor, self.z * factor)
        else:
            raise Exception('Operation not supported.')


def get_containing_orthonormal_basis(vector: Vector):
    """
    Gets 3 vectors that form an orthonormal basis in Euclidean space (including the normalized input vector)
    :param vector: first vector of the basis
    :return: 3 basis vectors
    """
    v1 = Vector(vector.x, vector.y, vector.z)   # create a copy of the input vector to normalize
    v1.normalize()  # normalize the vectors

    x, y, z = 0, 0, 0   # set components of second basis vector to zero
    eps = 0.00000001    # threshold for components to be considered zero
    zc = 0  # zero count
    for i in range(3):  # count the zero components in the vector
        if np.abs(v1[i]) < eps:
            zc += 1

    if zc >= 1:     # if any components of the input vector are zero, then the second basis vector is trivial to find
        if np.abs(v1.x) < eps:
            x = 1
        elif np.abs(v1.y) < eps:
            y = 1
        elif np.abs(v1.z) < eps:
            z = 1
    else:  # use the formula x1*x2 + y1*0 + z1*z2 = 0 to determine v2
        x = 1
        z = -v1.x / v1.z

    v2 = Vector(x, y, z)    # create second basis vector
    v2.normalize()      # normalize second basis vector
    v3 = cross(v1, v2)  # create third basis vector

    return v1, v2, v3


def get_rotation(vector: Vector):
    """
    Get the angles which the vector has with the Euclidean unit vectors
    :param vector: vector to check
    :return: 3 angles with the unit axes
    """
    unit_x = Vector(1, 0, 0)
    unit_y = Vector(0, 1, 0)
    unit_z = Vector(0, 0, 1)
    direction = Vector(vector.x, vector.y, vector.z)
    direction.normalize()
    theta_x = np.arccos(direction @ unit_x)
    theta_y = np.arccos(direction @ unit_y)
    theta_z = np.arccos(direction @ unit_z)
    return theta_x, theta_y, theta_z
