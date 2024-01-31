import numpy as np
from Geometry import is_number


def cross(a, b):
    """
    Returns the cross product of two vectors
    :param a: first vector
    :param b: second vector
    :return: cross product
    """
    return Vector.create_from_ndarray(np.cross(a._vector[:, 0], b._vector[:, 0]).reshape((3, 1)))


class Vector(object):
    """
    A vector with 3 components
    """

    @property
    def x(self):
        return self._vector[0, 0]

    @x.setter
    def x(self, value):
        self._vector[0, 0] = value

    @property
    def y(self):
        return self._vector[1, 0]

    @y.setter
    def y(self, value):
        self._vector[1, 0] = value

    @property
    def z(self):
        return self._vector[2, 0]

    @z.setter
    def z(self, value):
        self._vector[2, 0] = value

    def __init__(self, x=0, y=0, z=0):
        self._vector = np.ndarray((3, 1))
        self._vector[0, 0] = x
        self._vector[1, 0] = y
        self._vector[2, 0] = z

    @staticmethod
    def create_from_ndarray(array):
        return Vector(array[0, 0], array[1, 0], array[2, 0])

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
        return np.linalg.norm(self._vector, 2)**2

    def magnitude(self):
        """
        Returns the magnitude of the vector
        :return: Number representing the magnitude
        """
        return np.linalg.norm(self._vector, 2)

    def normalize(self):
        """
        Normalize the vectors magnitude to 1
        """
        self._vector /= self.magnitude()

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
        return np.any(np.isnan(self._vector))

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
        return self.create_from_ndarray(self._vector + other._vector)

    def __isub__(self, other):
        return self.create_from_ndarray(self._vector - other._vector)

    def __add__(self, other):
        return self.create_from_ndarray(self._vector + other._vector)

    def __sub__(self, other):
        return self.create_from_ndarray(self._vector - other._vector)

    def __neg__(self):
        return self.create_from_ndarray(-self._vector)

    def __matmul__(self, other):
        return np.dot(self._vector.T, other._vector)[0, 0]

    def __mul__(self, other):
        if is_number(other):
            return self.create_from_ndarray(other * self._vector)
        elif type(other) == Vector:
            return self @ other

    def __rmul__(self, other):
        if is_number(other):
            return self * other

    def __truediv__(self, other):
        if is_number(other):
            factor = 1 / other
            return self.create_from_ndarray(factor * self._vector)
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
