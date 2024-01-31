import unittest
import numpy as np
from Geometry import Vector, get_containing_orthonormal_basis, Ray, Matrix3x3, rotation_z_axis, rotation_y_axis, rotation_x_axis, identity


def get_test_vectors():
    a = Vector(1, 2, 3)
    b = Vector(5, 1, 3)
    return a, b


def get_test_matrices(count):
    matrices = list()
    m1 = Matrix3x3(2, 1, 1, 0, 2, 1, 0, 0, 1)
    m2 = Matrix3x3(0, 7, 1, 1, 2, 3, 3, 0, 1)
    m3 = Matrix3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    pre_defined = [m1, m2, m3]

    for i in range(count):
        matrices.append(pre_defined[i % len(pre_defined)])

    return tuple(matrices)


class TestMatrix(unittest.TestCase):

    def assert_matrix_equals(self, matrix, desired_matrix):
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(matrix[i, j], desired_matrix[i, j])

    def assert_vector_equals(self, vector, desired_vector):
        return self.assertAlmostEqual(vector.x, desired_vector.x) and self.assertAlmostEqual(vector.x, desired_vector.x) and self.assertAlmostEqual(vector.x, desired_vector.x)

    def test_determinant(self):
        m1, m2 = get_test_matrices(2)
        det1 = 4
        det2 = 49.99999999999999
        self.assertAlmostEqual(m1.determinant(), det1)
        self.assertAlmostEqual(m2.determinant(), det2)

    def test_add(self):
        m1, m2 = get_test_matrices(2)
        res = Matrix3x3(2, 8, 2, 1, 4, 4, 3, 0, 2)
        self.assert_matrix_equals(m1 + m2, res)
        self.assert_matrix_equals(m2 + m1, res)
        m1 += m2
        self.assert_matrix_equals(m1, res)

    def test_subtract(self):
        m1, m2 = get_test_matrices(2)
        res1 = Matrix3x3(2, -6, 0, -1, 0, -2, -3, 0, 0)
        res2 = Matrix3x3(-2, 6, 0, 1, 0, 2, 3, 0, 0)
        self.assert_matrix_equals(m1 - m2, res1)
        self.assert_matrix_equals(m2 - m1, res2)
        m1 -= m2
        self.assert_matrix_equals(m1, res1)

    def test_multiplication_with_number(self):
        m1 = get_test_matrices(1)[0]
        number = 7
        res = Matrix3x3(14, 7, 7, 0, 14, 7, 0, 0, 7)
        self.assert_matrix_equals(number * m1, res)
        self.assert_matrix_equals(m1 * number, res)

    def test_multiplication_with_vector(self):
        m1 = get_test_matrices(1)[0]
        v1, v2 = get_test_vectors()
        res1 = Vector(7, 7, 3)
        res2 = Vector(14, 5, 3)
        self.assert_vector_equals(m1 * v1, res1)
        self.assert_vector_equals(m1 * v2, res2)

    def test_multiplication_with_matrix(self):
        m1, m2 = get_test_matrices(2)
        res1 = Matrix3x3(4, 16,  6,  5,  4,  7,  3,  0,  1)
        res2 = Matrix3x3(0, 14,  8,  2,  5,  6,  6,  3,  4)
        self.assert_matrix_equals(m1 * m2, res1)
        self.assert_matrix_equals(m2 * m1, res2)

    def test_inversion(self):
        m1, m2 = get_test_matrices(2)
        res1 = Matrix3x3(0.5, -0.25, -0.25,  0.,  0.5, -0.5,  0.,  0.,  1.)
        res2 = Matrix3x3(0.04, -0.14,  0.38,  0.16, -0.06,  0.02, -0.12,  0.42, -0.14)
        id = identity()
        print(m1.inverse() * m1)
        self.assert_matrix_equals(m1.inverse(), res1)
        self.assert_matrix_equals(m2.inverse(), res2)
        self.assert_matrix_equals(m1.inverse() * m1, id)
        self.assert_matrix_equals(m2 * m2.inverse(), id)

    def test_transpose(self):
        m1, m2 = get_test_matrices(2)
        res1 = Matrix3x3(2, 0, 0, 1, 2, 0, 1, 1, 1)
        res2 = Matrix3x3(0, 1, 3, 7, 2, 0, 1, 3, 1)
        self.assert_matrix_equals(m1.T, res1)
        self.assert_matrix_equals(m2.T, res2)
        self.assert_matrix_equals(m1.T.T, m1)
        self.assert_matrix_equals(m2.T.T, m2)

    def test_rotation_x(self):
        m = rotation_x_axis(np.pi / 2)
        v1 = Vector(0, 0, 1)
        v2 = Vector(0, 1, 0)
        self.assert_vector_equals(m * v1, v2)

    def test_rotation_y(self):
        m = rotation_y_axis(np.pi / 2)
        v1 = Vector(0, 0, 1)
        v2 = Vector(-1, 0, 0)
        self.assert_vector_equals(m * v1, v2)

    def test_rotation_z(self):
        m = rotation_z_axis(np.pi / 2)
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        self.assert_vector_equals(m * v1, v2)


class TestRay(unittest.TestCase):

    def test_ray_propagation(self):
        ray = Ray(Vector(0, 0), Vector(76, 0))
        point1 = ray.get_point(10)
        self.assertAlmostEqual(point1.x, 10)
        self.assertAlmostEqual(point1.y, 0)

        ray = Ray(Vector(0, 0), Vector(0, -5))
        point1 = ray.get_point(10)
        self.assertAlmostEqual(point1.x, 0)
        self.assertAlmostEqual(point1.y, -10)


class TestVector(unittest.TestCase):

    def assert_vector_equality(self, a, b):
        self.assertAlmostEqual(a.x, b.x)
        self.assertAlmostEqual(a.y, b.y)
        self.assertAlmostEqual(a.z, b.z)

    def test_adding(self):
        a, b = get_test_vectors()
        res = Vector(a.x + b.x, a.y + b.y, a.z + b.z)
        self.assert_vector_equality(a + b, res)
        self.assert_vector_equality(b + a, res)

    def test_subtraction(self):
        a, b = get_test_vectors()
        res1 = Vector(a.x - b.x, a.y - b.y, a.z - b.z)
        res2 = Vector(b.x - a.x, b.y - a.y, b.z - a.z)
        self.assert_vector_equality(a - b, res1)
        self.assert_vector_equality(b - a, res2)

    def test_multiplication(self):
        a, b = get_test_vectors()
        c = 77.2
        res1 = a.x * b.x + a.y * b.y + a.z * b.z
        res2 = Vector(c * a.x, c * a.y, c * a.z)
        self.assertEqual(a * b, res1)
        self.assertEqual(b * a, res1)
        self.assert_vector_equality(a * c, res2)
        self.assert_vector_equality(c * a, res2)

    def test_dot_product(self):
        a, b = get_test_vectors()
        res1 = a.x * b.x + a.y * b.y + a.z * b.z
        self.assertEqual(a @ b, res1)
        self.assertEqual(b @ a, res1)

    def test_division(self):
        a, b = get_test_vectors()
        div1 = 7
        div2 = 2.1
        res1 = Vector(a.x / div1, a.y / div1, a.z / div1)
        res2 = Vector(b.x / div2, b.y / div2, b.z / div2)
        self.assert_vector_equality(a / div1, res1)
        self.assert_vector_equality(b / div2, res2)

    def test_vector_coordinate_system(self):
        v1, v2, v3 = get_containing_orthonormal_basis(Vector(2, 1, 5))
        self.assertAlmostEqual(v1 @ v2, 0)
        self.assertAlmostEqual(v1 @ v3, 0)
        self.assertAlmostEqual(v2 @ v3, 0)


if __name__ == '__main__':
    unittest.main()
