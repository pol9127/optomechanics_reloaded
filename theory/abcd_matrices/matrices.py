import numpy as np
from matplotlib.patches import Polygon
from optomechanics.theory.abcd_matrices.math import refractive_index_to_color
class ABCD:
    def __init__(self):
        self.matrices = {}
        self.width = 0

    def calc_matrix(self, A, B, C, D, name):
        self.matrices[name] = np.array([[A, B],
                                        [C, D]])

    def __mul__(self, other):
        names = list(self.matrices.keys()) + list(set(list(other.matrices.keys())) - set(list(self.matrices.keys())))
        for name in names:
            if name in list(self.matrices.keys()) and name in list(other.matrices.keys()):
                self.matrices[name] = self.matrices[name] @ other.matrices[name]
            elif name in list(other.matrices.keys()):
                self.matrices[name] = other.matrices[name]


class Medium(ABCD):
    def __init__(self, distance, n_final=1, height=10e-3, pos=0, patch_offset=0, n_range=[1, 5]):
        super().__init__()
        self.calc_matrix(1, distance, 0, 1, 'transmission')
        self.calc_matrix(1, distance, 0, 1, 'reflection')
        self.calc_matrix(1, distance, 0, 1, 'reverse_transmission')
        self.tags = {'distance' : distance, 'n_final' : n_final}
        self.height = height
        self.pos = pos + patch_offset
        self.n_range = n_range

    def __repr__(self):
        return 'medium'

    def update_distance(self, new_distance):
        self.calc_matrix(1, new_distance, 0, 1, 'transmission')
        self.calc_matrix(1, -1 * new_distance, 0, 1, 'reflection')
        self.tags['distance'] = new_distance

    def update_n_final(self, new_n_final):
        self.calc_matrix(1, self.tags['distance'], 0, 1, 'transmission')
        self.calc_matrix(1, -1 * self.tags['distance'], 0, 1, 'reflection')
        self.tags['n_final'] = new_n_final

    @property
    def patch(self):
        points = np.array([[self.pos, -0.5 * self.height],
                           [self.tags['distance'] + self.pos, -0.5 * self.height],
                           [self.tags['distance'] + self.pos, 0.5 * self.height],
                           [self.pos, 0.5 * self.height]])
        patch_ = Polygon(points, facecolor=refractive_index_to_color(self.tags['n_final'], n_range=self.n_range),
                         edgecolor=refractive_index_to_color(self.tags['n_final'], n_range=self.n_range))
        return patch_


class FlatSurface(ABCD):
    def __init__(self, n_initial, n_final, height=10e-3, pos=0, n_range=[1, 5]):
        super().__init__()
        self.calc_matrix(1, 0, 0, n_initial / n_final, 'transmission')
        self.calc_matrix(1, 0, 0, 1, 'reflection')
        self.calc_matrix(1, 0, 0, n_final / n_initial, 'reverse_transmission')
        self.tags = {'n_initial' : n_initial, 'n_final' : n_final}
        self.height = height
        self.pos = pos
        self.n_range = n_range


    def __repr__(self):
        return 'flat surface'

    def update_n_initial(self, new_n_initial):
        self.calc_matrix(1, 0, 0, new_n_initial / self.tags['n_final'], 'transmission')
        self.tags['n_initial'] = new_n_initial

    def update_n_final(self, new_n_final):
        self.calc_matrix(1, 0, 0, self.tags['n_initial'] / new_n_final, 'transmission')
        self.tags['n_final'] = new_n_final

    @property
    def patch(self):
        points = np.array([[self.pos, -0.5 * self.height],
                           [self.pos, 0.5 * self.height]])
        patch_ = Polygon(points, edgecolor=refractive_index_to_color(self.tags['n_final'], n_range=self.n_range))
        return patch_


class CurvedSurface(ABCD):
    def __init__(self, n_initial, n_final, curvature, height=10e-3, pos=0, n_range=[1, 5]):
        super().__init__()
        self.calc_matrix(1, 0, (n_initial - n_final) / (curvature * n_final), n_initial / n_final, 'transmission')
        self.calc_matrix(1, 0, 2 / curvature, 1, 'reflection')
        self.calc_matrix(1, 0, (n_final - n_initial) / (-1 * curvature * n_initial), n_final / n_initial, 'reverse_transmission')
        self.tags = {'n_initial' : n_initial, 'n_final' : n_final, 'curvature' : curvature}
        self.n_range = n_range

        if height / 2 > abs(curvature):
            self.height = abs(curvature)
        else:
            self.height = height / 2

        if curvature > 0:
            self.width = abs(curvature) * (1 - abs(np.cos(np.arcsin(self.height / curvature))))
        else:
            self.width = 0
        self.pos = pos


    def __repr__(self):
        return 'curved surface'

    def update_n_initial(self, new_n_initial):
        self.calc_matrix(1, 0, (new_n_initial - self.tags['n_final']) / (self.tags['curvature'] * self.tags['n_final']), new_n_initial / self.tags['n_final'], 'transmission')
        self.tags['n_initial'] = new_n_initial

    def update_n_final(self, new_n_final):
        self.calc_matrix(1, 0, (self.tags['n_initial'] - new_n_final) / (self.tags['curvature'] * new_n_final), self.tags['n_initial'] / new_n_final, 'transmission')
        self.tags['n_final'] = new_n_final

    def update_curvature(self, new_curvature):
        self.calc_matrix(1, 0, (self.tags['n_initial'] - self.tags['n_final']) / (new_curvature * self.tags['n_final']), self.tags['n_initial'] / self.tags['n_final'], 'transmission')
        self.calc_matrix(1, 0, -2 / new_curvature, 1, 'reflection')
        self.tags['curvature'] = new_curvature

    @property
    def patch(self, pixel=10000):
        phi_lim = np.arcsin(self.height / self.tags['curvature'])

        phi_points = np.linspace(-phi_lim, phi_lim, pixel)

        points = -1*self.tags['curvature'] * np.array([np.cos(phi_points) - 1, np.sin(phi_points)]).T
        if self.tags['curvature'] < 0:
            points_1 = np.vstack((np.array([0, points[0, 1]]), points, np.array([0, points[-1, 1]])))
            points_2 = points
        else:
            points_2 = np.vstack((np.array([0, points[0, 1]]), points, np.array([0, points[-1, 1]])))
            points_1 = points


        points_2[:, 0] += self.pos
        points_1[:, 0] += self.pos

        patch_1 = Polygon(points_1, facecolor=refractive_index_to_color(self.tags['n_final'], n_range=self.n_range),
                          edgecolor=refractive_index_to_color(self.tags['n_final'], n_range=self.n_range))
        patch_2 = Polygon(points_2, facecolor=refractive_index_to_color(self.tags['n_initial'], n_range=self.n_range),
                          edgecolor=refractive_index_to_color(self.tags['n_initial'], n_range=self.n_range))
        return patch_1, patch_2
