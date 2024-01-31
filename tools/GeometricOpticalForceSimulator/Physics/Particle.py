from Geometry import Sphere, Vector


class Particle(Sphere):
    """
    A particle which can interact with light

    By default the particle will have the refractive index of SiO2 as well as its density
    """

    @property
    def mass(self):
        """
        Gets the mass of the particle
        """
        return self.volume * self.density

    def __init__(self, n=1.59, density=1850, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.force = Vector()
        self.scattering_force = Vector()
        self.gradient_force = Vector()
        self.density = density
        self._mass = 0

    def get_force_along(self, direction: Vector):
        """
        Returns the optical force acting on the particle along a specified axis
        :param direction: axis
        :return: force acting on the particle along the given axis
        """
        direction.normalize()
        return self.force @ direction
