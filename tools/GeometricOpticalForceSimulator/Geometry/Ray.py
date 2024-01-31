from Geometry import Vector


class Ray(object):
    """
    A ray that can be used to probe for intersections in a specified path
    """

    def __init__(self, origin: Vector, direction: Vector):
        self.origin = origin
        self.direction = direction
        self.direction.normalize()

    def get_point(self, t: float):
        """
        Get a point on the ray
        :param t: distance from ray origin
        :return: vector that represents the point at the specified distance on the ray from the origin
        """
        return self.origin + t * self.direction
