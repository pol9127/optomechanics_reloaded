from Geometry import Ray, Vector, cross
from Physics.Fresnesl import transmission, reflection
import numpy as np

EPSILON = 1e-12


class LightRay(Ray):
    """
    A light ray used to simulate optical forces
    """

    def __init__(self, wavelength=1565e-9, power=1, reflection_limit=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wavelength = wavelength
        self.power = power
        self.reflection_limit = reflection_limit
        self.has_end_point = False
        self.was_reflected = False
        self.was_refracted = False
        self.end_point = None

    def get_angles_at_interface(self, plane_normal: Vector, n1: float, n2: float):
        """
        Returns the incident and transmitted angles of this ray when hitting an interface
        :param plane_normal: normal vector of the interface at the hit point
        :param n1: refractive index in the material before the interface
        :param n2: refractive index in the material after the interface
        :return: in angle and out angle
        """
        theta_in = np.arccos(- self.direction @ plane_normal)
        theta_out = np.arcsin(n1 / n2 * np.sin(theta_in))
        return theta_in, theta_out

    def get_reflected_ray(self, n1: float, n2: float, point: Vector, normal: Vector):
        """
        Returns a new ray that is spawned by a reflection at an interface
        :param n1: refractive index in the material before the interface
        :param n2: refractive index in the material after the interface
        :param point: point where the interface was hit
        :param normal: normal vector of the interface at the hit point
        :return: reflected ray
        """
        theta_i, theta_o = self.get_angles_at_interface(normal, n1, n2)     # get angles before/after interface
        r = reflection(n1, n2, theta_i, theta_o)    # get the Fresnel coefficient (circular / no polarization)

        # get unit vectors which make up reflected ray
        basis_1 = normal    # first unit vector is the normal
        basis_2 = cross(normal, cross(normal, self.direction))  # combination of ray direction and normal

        # only try to normalize the second unit vector if it is not zero
        # (can be zero if ray direction si co-linear with normal)
        if basis_2.magnitude() > EPSILON:
            basis_2.normalize()

        # calculate the new direction and origin of the reflected ray
        new_direction = basis_1 * np.cos(theta_i) - basis_2 * np.sin(theta_i)
        origin = point + new_direction * EPSILON    # slightly move away from the surface to avoid accidental reflection

        # create the reflected ray with attenuated power and new propagation direction and origin
        reflected_ray = LightRay(self.wavelength, self.power * r, self.reflection_limit - 1, origin=origin, direction=new_direction)
        reflected_ray.was_reflected = True     # the new ray knows it was the result of a reflection
        reflected_ray.was_refracted = self.was_refracted   # keep the memory about past refractions
        return reflected_ray

    def get_transmitted_ray(self, n1: float, n2: float, point: Vector, normal: Vector):
        """
        Returns a new ray that is spawned by a transmission through an interface
        :param n1: refractive index in the material before the interface
        :param n2: refractive index in the material after the interface
        :param point: point where the interface was hit
        :param normal: normal vector of the interface at the hit point
        :return: transmitted ray
        """
        theta_i, theta_o = self.get_angles_at_interface(normal, n1, n2)     # get angles before/after interface
        t = transmission(n1, n2, theta_i, theta_o)    # get the Fresnel coefficient (circular / no polarization)

        # get unit vectors which make up reflected ray
        basis_1 = normal  # first unit vector is the normal
        basis_2 = cross(normal, cross(normal, self.direction))  # combination of ray direction and normal

        # only try to normalize the second unit vector if it is not zero
        # (can be zero if ray direction si co-linear with normal)
        if basis_2.magnitude() > EPSILON:
            basis_2.normalize()

        # calculate the new direction and origin of the transmitted ray
        new_direction = -basis_1 * np.cos(theta_o) - basis_2 * np.sin(theta_o)
        origin = point + new_direction * EPSILON    # slightly move away from the surface to avoid accidental reflection

        # create the new transmitted ray with attenuated power and new direction as well as origin
        transmitted_ray = LightRay(self.wavelength, self.power * t, self.reflection_limit, origin=origin, direction=new_direction)
        transmitted_ray.was_reflected = self.was_reflected     # keep the memory of previous reflections
        transmitted_ray.was_refracted = True   # the ray knows that it was the result of a transmission
        return transmitted_ray
