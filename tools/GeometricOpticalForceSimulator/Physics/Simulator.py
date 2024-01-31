import numpy as np
import matplotlib.pyplot as plt
"""from mayavi import mlab
from tvtk.tools import visual"""
from matplotlib.colors import to_rgba
from Physics import Particle, Medium, LightRay
from Physics.GaussianBeam import GaussianBeam, generate_gaussian_beam
from Geometry import Vector, get_containing_orthonormal_basis


class Simulator(object):
    """
    A Simulator that uses ray tracing to simulate optical forces on physical objects
    """

    def __init__(self):
        self.beam_color = 'r'
        self.c = 299792458
        self.particles = list()
        self.medium = Medium()
        self.rays = list()
        self.beam_direction = Vector(1, 0, 0)
        self._simulated_rays = list()
        self._simulated_force_deltas = list()
        self.beam = None
        self.max_recursion_depth = 100

    def simulate_displacement(self, particle: Particle, axis: Vector, displacements):
        """
        Simulate the displacement of a given particle along a specified axis
        :param particle: particle to simulate the displacement on
        :param axis: axis to displace the particle on
        :param displacements: displacements of the particle on the given axis
        :return: displacement, gradient forces and scattering force
        """

        # if the particle is not yet present in the simulation domain add it
        if particle not in self.particles:
            self.particles.append(particle)

        print('simulating particle displacement...')

        # get local coordinate system of the beam
        unit_x, unit_y, unit_z = get_containing_orthonormal_basis(self.beam_direction)

        axial_force = list()
        radial_force_y = list()
        radial_force_z = list()
        gradiant_force = list()
        scattering_force = list()
        step_count = len(displacements)
        step = 0
        initial_particle_position = Vector(particle.position.x, particle.position.y, particle.position.z)

        # simulate the displacement
        for d in displacements:
            particle.position = initial_particle_position + axis * d    # set particle position
            self.simulate(verbose=False)    # simulate the forces
            axial_force.append(particle.get_force_along(unit_x))
            radial_force_y.append(particle.get_force_along(unit_y))
            radial_force_z.append(particle.get_force_along(unit_z))
            gradiant_force.append(particle.gradient_force @ particle.gradient_force.normalized())
            scattering_force.append(particle.scattering_force @ particle.scattering_force.normalized())
            step += 1
            print('Progress: {} %'.format(np.round(step / step_count * 100, 2)))

        particle.position = initial_particle_position

        return displacements, np.array(radial_force_y), np.array(radial_force_z), np.array(axial_force), np.array(gradiant_force), np.array(scattering_force)

    def simulate(self, verbose=True):
        """
        Simulate light matter interactions in the simulation domain
        :param verbose: if true print statements about the current simulation step
        """
        # remove all reflected and transmitted rays of previous simulations
        self._simulated_rays.clear()
        self._simulated_force_deltas.clear()

        # set all forces acting on particles to zero
        for particle in self.particles:
            particle.force = Vector()
            particle.scattering_force = Vector()
            particle.gradient_force = Vector()

        i = 0
        # trace all rays in the simulation domain through the environment
        for ray in self.rays:
            ray.has_end_point = False  # reset the endpoint
            force, hit_count = self.trace_ray(ray)    # trace the ray

            if verbose:
                print('traced ray {}: total hits={}, total force={} pN'.format(i, hit_count, force.magnitude()*1e12))
                i += 1

        # displace the optical forces acting on the particles if verbose is set to true
        if verbose:
            i = 0
            for particle in self.particles:
                print('gradient force on particle {}: {} pN'.format(i, particle.gradient_force*1e12))
                print('scattering force on particle {}: {} pN'.format(i, particle.scattering_force*1e12))
                i += 1

    def trace_ray(self, ray: LightRay, recursion_depth=0):
        """
        Trace a ray through the simulation domain.
        :param ray: ray to trace
        :param recursion_depth: current recursion depth (status variable)
        :return: force exerted by this ray, hit count (interfaces hit during tracing)
        """
        total_hits = 0
        total_force = Vector()

        # cancel ray tracing if the maximum recursion depth has been reached
        if recursion_depth >= self.max_recursion_depth:
            return total_force, total_hits

        # check all particles in the domain for intersections with the ray
        # Find particle which has the intersection closest to the rays origin
        hit = False
        hit_particle = None
        hit_point = None
        hit_normal = None
        hit_distance = 0
        for particle in self.particles:
            hit, point, normal = particle.intersect(ray)

            if hit:
                distance = (ray.origin - point).magnitude()
                if hit_point is None or distance > hit_distance:
                    hit_distance = distance
                    hit_normal = normal
                    hit_point = point
                    hit_particle = particle

        # if a particle was hit, handle reflection and refraction of the ray
        if hit:
            total_hits += 1     # add to the total hits of the ray
            ray.has_end_point = True   # set an endpoint for the ray
            ray.end_point = hit_point

            n1 = self.medium.n      # get refractive index from environment
            n2 = hit_particle.n     # get refractive index of the particle

            # swap the refractive indices if the rays origin lies within the particle
            if hit_particle.contains(ray.origin):
                n1 = hit_particle.n
                n2 = self.medium.n

            force_ref = Vector()    # initialize zero force for collecting all influences
            force_in = (ray.power * n1 / self.c) * ray.direction
            total_force = force_in

            # spawn the transmitted ray
            transmitted_ray = ray.get_transmitted_ray(n1, n2, hit_point, hit_normal)
            force_trans = (transmitted_ray.power * n2 / self.c) * transmitted_ray.direction
            total_force -= force_trans

            # add the transmitted ray to the simulation result and trace it
            self._simulated_rays.append(transmitted_ray)
            force, hit_count = self.trace_ray(transmitted_ray, recursion_depth + 1)
            total_hits += hit_count
            total_force += force

            # if the reflection limit has not yet been exceeded also handle the reflected ray
            if ray.reflection_limit > 0:
                # spawn the reflected ray
                reflected_ray = ray.get_reflected_ray(n1, n2, hit_point, hit_normal)
                force_ref = (reflected_ray.power * n1 / self.c) * reflected_ray.direction
                total_force -= force_ref

                # add the reflected ray to the simulation result and trace it
                self._simulated_rays.append(reflected_ray)
                force, hit_count = self.trace_ray(reflected_ray, recursion_depth + 1)
                total_hits += hit_count
                total_force += force

            # Calculate the force exerted from this ray by the change in photon momentum on hitting the particle
            ray_force = force_in - force_trans - force_ref
            ray_scattering_force = (ray_force @ ray.direction) * ray.direction
            ray_gradient_force = ray_force - ray_scattering_force
            hit_particle.force += ray_force
            hit_particle.scattering_force += ray_scattering_force
            hit_particle.gradient_force += ray_gradient_force

            # add force delta for field generation
            self._simulated_force_deltas.append([hit_point, force_in - force_trans - force_ref])

        return total_force, total_hits

    def setup_beam_gaussian_3d(self, waist_position: Vector, direction: Vector, wavelength=1565e-9, power=1, numerical_aperture=1, n_diagonal_beams=30, reflection_limit=5, far_field=False, diverging_near_field=False):
        """
        Add Gaussian beam to the simulation domain
        :param waist_position: position of the focus
        :param direction: direction of the beam
        :param wavelength:  wavelength of the beam
        :param power: power carried by the beam (W)
        :param numerical_aperture: numerical aperture of the beam
        :param n_diagonal_beams: diagonal beam count. (real beam count n = nd * pi/(2w0)) -> w0 beam waist
        :param reflection_limit: maximum number of reflections permitted (controls simulation performance)
        """
        beam_waist = wavelength / (numerical_aperture * np.pi)
        self.beam_direction = direction
        self.rays, self.beam = generate_gaussian_beam(waist_position, direction, wavelength, beam_waist, power, n_diagonal_beams, reflection_limit, far_field, diverging_near_field)

    def setup_beam_gaussian_2d(self, wavelength=1565e-9, power=1, numerical_aperture=1, n_beams=30, reflection_limit=5):
        """
        Add a planar projection of a Gaussian beam into the simulation domain
        :param wavelength: wavelength of the beam
        :param power: power carried by the beam
        :param numerical_aperture: numerical aperture of the beam
        :param n_beams: number of beams used to simulate the Gaussian beam
        :param reflection_limit: maximum number of reflections permitted (controls simulation performance)
        """
        self.rays.clear()
        beam_waist = wavelength / (numerical_aperture * np.pi)
        self.beam = GaussianBeam(wavelength, beam_waist, power)

        beam_area = np.pi * beam_waist**2
        delta_area = beam_area / n_beams

        origin_x = -self.beam.rayleigh_range
        a = 2 * self.beam.waist(origin_x)
        da = a / n_beams
        a0 = a / 2
        b = 2 * beam_waist
        db = b / n_beams
        b0 = b / 2

        for i in range(n_beams):
            origin_y = a0 - i * da
            direction = Vector(self.beam.rayleigh_range, b0 - i * db - origin_y)
            beam_power = self.beam.intensity_at(0, b0 - i * db) * delta_area
            beam = LightRay(wavelength, beam_power, reflection_limit, origin=Vector(origin_x, origin_y), direction=direction)
            self.rays.append(beam)

    def visualize_2d(self, show_reflected_beams=True,
                     show_beams=True,
                     use_intensity_for_visibility=True,
                     x_axis=Vector(1, 0, 0),
                     y_axis=Vector(0, 0, 1),
                     title=None,
                     xlabel='x',
                     ylabel='y',
                     center_particle: Particle=None,
                     figzise=(2, 1),
                     show=True,
                     save_path=None,
                     view_box=None,
                     show_force_field=False,
                     field_grid_size=2e-6,
                     particle_color='b'):
        """
        Use plotting to display the simulation domain projected onto a given set of axes.
        :param show_reflected_beams: If true, reflected beams will be displayed
        :param x_axis: world direction of the plot x axis
        :param y_axis: world direction of the plot y axis
        :param title: title of the plot
        :param xlabel: x label of the plot
        :param ylabel: y label of the plot
        :param figzise: The relative size of the figure
        :param show: displays the figure
        :param save_path: if set to something else than None, will save the figure
        :param center_particle: particle on which the view should be centered
        :param view_box: view box which should be used for the view
        :param show_force_field: show vector quivers to display the force field
        :param particle_color: color of the particles in the plot
        :param field_grid_size: grid size of field
        """
        x_axis.normalize()
        y_axis.normalize()

        (br, bg, bb, ba) = to_rgba(self.beam_color)
        um = 1e6
        pN = 1e12
        fig, ax = plt.subplots()
        fig.set_figwidth = figzise[0]
        fig.set_figheight = figzise[1]
        gca = plt.gca()

        x_limits = [-20e-6, 20e-6]
        y_limits = [-10e-6, 10e-6]

        max_beam_power = 0
        for beam in self.rays:
            if beam.power > max_beam_power:
                max_beam_power = beam.power

        if view_box is not None:
            x_limits = [view_box[0], view_box[0] + view_box[2]]
            y_limits = [view_box[1], view_box[1] + view_box[3]]
        elif self.beam is not None:
            factor = figzise[1] / figzise[0]
            x_limits = [-self.beam.rayleigh_range, self.beam.rayleigh_range]
            y_limits = [-self.beam.rayleigh_range * factor, self.beam.rayleigh_range * factor]

        if center_particle is not None:
            x_offset = center_particle.position @ x_axis
            y_offset = center_particle.position @ y_axis
            x_limits[0] = x_limits[0] + x_offset
            x_limits[1] = x_limits[1] + x_offset
            y_limits[0] = y_limits[0] + y_offset
            y_limits[1] = y_limits[1] + y_offset

        for particle in self.particles:
            x = particle.position @ x_axis
            y = particle.position @ y_axis
            circle = plt.Circle([x*um, y*um], particle.radius*um, fc=particle_color)
            gca.add_patch(circle)

        if show_beams:
            for beam in self.rays:
                end_point = beam.end_point if beam.has_end_point else beam.get_point(10)
                x1 = beam.origin @ x_axis
                y1 = beam.origin @ y_axis
                x2 = end_point @ x_axis
                y2 = end_point @ y_axis
                if use_intensity_for_visibility:
                    beam_color = (br, bg, bb, ba * beam.power / max_beam_power)
                else:
                    beam_color = (br, bg, bb, 1)
                line = plt.Line2D((x1*um, x2*um), (y1*um, y2*um), c=beam_color, lw=1)
                gca.add_line(line)

            for beam in self._simulated_rays:
                if not show_reflected_beams and beam.was_reflected:
                    continue
                end_point = beam.end_point if beam.has_end_point else beam.get_point(10)
                x1 = beam.origin @ x_axis
                y1 = beam.origin @ y_axis
                x2 = end_point @ x_axis
                y2 = end_point @ y_axis
                if use_intensity_for_visibility:
                    beam_color = (br, bg, bb, ba * beam.power / max_beam_power)
                else:
                    beam_color = (br, bg, bb, 1)
                line = plt.Line2D((x1 * um, x2 * um), (y1 * um, y2 * um), c=beam_color, lw=1)
                gca.add_line(line)

        ax.set_aspect(1)
        ax.set_xlabel('{} [µm]'.format(xlabel))
        ax.set_ylabel('{} [µm]'.format(ylabel))

        if show_force_field:
            dx = field_grid_size
            x_grid = np.arange(x_limits[0], x_limits[1], dx)
            y_grid = np.arange(y_limits[0], y_limits[1], dx)
            X, Y = np.meshgrid(x_grid, y_grid)
            U, V = np.zeros(X.shape), np.zeros(Y.shape)

            hit_points, hit_forces = self.to_numpy(self._simulated_force_deltas)

            unit_x = np.array([x_axis.x, x_axis.y, x_axis.z]).reshape([1, 3])
            unit_y = np.array([y_axis.x, y_axis.y, y_axis.z]).reshape([1, 3])
            point_2d_x = np.matmul(unit_x, hit_points.T).ravel()  # dot product each point with x axis -> x coordinates in 2D
            point_2d_y = np.matmul(unit_y, hit_points.T).ravel()  # dot product each point with y axis -> y coordinates in 2D
            force_2d_x = np.matmul(unit_x, hit_forces.T).ravel()  # dot product each force with x axis -> x coordinates in 2D
            force_2d_y = np.matmul(unit_y, hit_forces.T).ravel()  # dot product each force with y axis -> y coordinates in 2D

            for i in range(len(point_2d_x)):
                x_arg = np.argmin(np.abs(x_grid - point_2d_x[i]))
                y_arg = np.argmin(np.abs(y_grid - point_2d_y[i]))
                U[y_arg, x_arg] += force_2d_x[i]
                V[y_arg, x_arg] += force_2d_y[i]

            M = np.hypot(U, V)
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    val1 = U[i, j]
                    val2 = V[i, j]
                    nrm = np.sqrt(val1**2 + val2**2)
                    if nrm > 1e-30:
                        U[i, j] /= nrm
                        V[i, j] /= nrm

            M_norm = M / np.max(M) + 0.5
            Q = plt.quiver(X*um, Y*um, U*30*M_norm, V*30*M_norm, M_norm, units='x', pivot='tip', width=0.3, scale=1 / 0.15)
            Q.set_zorder(100)
            """qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                               coordinates='figure')"""
            #plt.scatter(X*um, Y*um, color='0.5', s=1)
        x_limits = [x_limits[0] * um, x_limits[1] * um]
        y_limits = [y_limits[0] * um, y_limits[1] * um]

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        if title is not None:
            ax.set_title(title)

        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def to_numpy(vec_list):
        positions = None
        forces = None
        for v in vec_list:
            if positions is None:
                positions = np.array([v[0].x, v[0].y, v[0].z])
            else:
                positions = np.vstack([positions, np.array([v[0].x, v[0].y, v[0].z])])
            if forces is None:
                forces = np.array([v[1].x, v[1].y, v[1].z])
            else:
                forces = np.vstack([forces, np.array([v[1].x, v[1].y, v[1].z])])
        return positions, forces

    def visualize_3d(self, show_reflected_beams=True):
        um = 1e6
        (br, bg, bb, ba) = to_rgba(self.beam_color)
        """mlab.clf()

        max_beam_power = 0
        for beam in self.beams:
            if beam.power > max_beam_power:
                max_beam_power = beam.power

        # Plot particles
        for particle in self.particles:
            radius = particle.radius * um
            x, y, z = np.mgrid[-radius:radius:50j, -radius:radius:50j, -radius:radius:50j]
            hull = (x - particle.position.x * um)**2 + (y - particle.position.y * um)**2 + (z - particle.position.z * um)**2 - radius**2
            mlab.contour3d(x, y, z, hull, contours=[0])

        for beam in self.beams:
            end_point = beam.end_point if beam.has_end_point else beam.get_point(10)
            radius = beam.power / max_beam_power * 0.5
            beam_color = (br, bg, bb)
            x_coord = [beam.origin.x, end_point.x]
            y_coord = [beam.origin.y, end_point.y]
            z_coord = [beam.origin.z, end_point.z]
            mlab.plot3d(x_coord, y_coord, z_coord, color=beam_color, tube_radius=radius)

        mlab.axes()
        mlab.show()"""


if __name__ == '__main__':
    sim = Simulator()
    sim.setup_beam_gaussian_3d(Vector(), Vector(0, 0, 1), numerical_aperture=0.037)
    print(sim.beam.numerical_aperture)
