from Geometry import Vector, Ray
import numpy as np


class Sphere(object):
    """
    A sphere
    """

    @property
    def radius(self):
        """
        Gets the radius of the sphere
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        """
        Sets the radius of the sphere
        :param value: new radius
        """
        self._radius = value
        self._volume = 4/3*np.pi*self._radius**3

    @property
    def volume(self):
        """
        Gets the volume of the sphere
        """
        return self._volume

    def __init__(self, position: Vector, radius: float):
        self.position = position
        self._radius = 0
        self._volume = 0
        self.radius = radius
        self._r2 = radius**2

    def contains(self, point: Vector):
        """
        Returns true if the points given lies within the sphere
        :param point: Point which should be checked
        :return: true if point is inside sphere
        """
        return (point - self.position).magnitude() < self.radius

    def intersect(self, ray: Ray):
        """
        Returns details about the first intersection of a ray with the sphere.
        :param ray: Ray to check for intersection
        :return: True or false if intersection is happening, the intersection point and the surface normal
        """

        # a scheme is followed where a triangle between two potential
        # intersection points and the sphere origin is evaluated
        delta = self.position - ray.origin  # get the offset of the ray and the sphere origin
        t_ca = delta @ ray.direction    # get the distance from the ray origin to the center of the long triangle side

        d2 = delta.magnitude_sqrt() - t_ca**2   # get the distance of the ray to the sphere center

        if d2 > self._r2:   # if the distance exceeds the radius the ray misses the sphere
            return False, None, None

        t_hc = np.sqrt(self._r2 - d2)   # get half the length of the triangle side opposite of the sphere center
        t0 = t_ca + t_hc    # first intersection point (distance from ray origin)
        t1 = t_ca - t_hc    # second intersection point (distance from ray origin)

        if t0 > t1:     # get intersection point that lies closer to the sphere origin
            temp = t0
            t0 = t1
            t1 = temp

        if t0 < 0:      # only use points in front of the ray origin
            t0 = t1
            if t0 < 0:      # if the sphere lies behind the ray origin it cannot intersect
                return False, None, None

        point = ray.get_point(t0)       # get the hit point in 3D space
        normal = point - self.position  # get the hit normal
        normal = normal / normal.magnitude()

        if point.is_nan() or normal.is_nan():   # if the numbers are not valid we say that there is no intersection
            return False, None, None

        if self.contains(ray.origin):   # if the origin of the ray lies within the sphere make normal pointing inward
            normal = -normal

        return True, ray.get_point(t0), normal
