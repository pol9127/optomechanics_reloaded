import sympy as sy
from scipy import constants

sy.init_printing()

class Framework(object):
    symbols = {}
    functions = {}

    def __init__(self, *libraries):
        self.libraries = {cl.__name__ : None for cl in self.__class__.__bases__[0].__subclasses__()}
        for lib in libraries:
            self.libraries[type(lib).__name__] = lib
        self.libraries[type(self).__name__] = self
        self.symbols_initialized = None
        self.definitions = None
        self.equations = None

        self.symbols_initialized = {smb : sy.symbols(**(self.symbols[smb][0])) for smb in self.symbols}
        vars(self).update(self.symbols_initialized)

        self.functions_initialized = {func : sy.Function(self.functions[func][0]['names'])(*[self.symbols_initialized[sym]
                                                                                            for sym in self.functions[func][0]['vars']])
                                      for func in self.functions}
        vars(self).update(self.functions_initialized)

        self.define_quantities()
        vars(self).update({df + '_' : self.definitions[df] for df in self.definitions})

        self.equations = [sy.Eq(self.symbols_initialized[func], self.definitions[func]) for func in self.definitions]


class Geometry(Framework):
    symbols = {'Vsp' : [{'names' : 'V_\\mathrm{sp}', 'positive' : True}, 'Volume of a sphere'],
               'Asp': [{'names': 'A_\\mathrm{sp}', 'positive': True}, 'Surface Area of a sphere'],
               'a': [{'names': 'a', 'positive': True}, 'length']
               }

    def __init__(self, *libraries):
        if not libraries:
            super().__init__(Constants())
        else:
            super().__init__(*libraries)

    def define_quantities(self):
        self.definitions = {'Vsp' : 4 / 3 * sy.pi * self.a ** 3,
                            'Asp': 4 * sy.pi * self.a ** 2
                            }


class Differential(Framework):
    symbols = {'EOM_harmonic' : [{'names' : '\\mathrm{EOM}_\\mathrm{harm}'}, 'Equation of Motion of the harmonic oscillator'],
               'alpha': [{'names': '\\alpha', 'real' : True}, 'Duffing Coefficient'],
               'Omega': [{'names': '\\Omega', 'real': True}, 'Oscillation Frequency'],
               'omega': [{'names': '\\omega', 'real': True}, 'Driving Frequency'],
               'omegaP': [{'names': '\\omega^\prime', 'real': True}, 'Driving Frequency'],
               'm': [{'names': 'm', 'real': True}, 'Mass'],
               'gamma': [{'names': '\\gamma', 'real': True}, 'Damping'],
               'lamda': [{'names': '\\lambda', 'real': True}, 'Parametric Driving Strength'],
               'theta': [{'names': '\\theta', 'real': True}, 'Phase'],
               't': [{'names': 't', 'real': True}, 'time'],
               'EOM_duffing': [{'names': '\\mathrm{EOM}_\\mathrm{harm}'}, 'Equation of Motion of the Duffing oscillator'],
               'EOM_matthieu': [{'names': '\\mathrm{EOM}_\\mathrm{harm}'}, 'Mathieu Equation']
               }

    functions = {'x': [{'names': 'x', 'vars' : ('t')}, 'Spatial Coordinate in an EOM'],
                 'F': [{'names': 'F', 'vars': ('t')}, 'Direct Driving Force']
                 }

    def __init__(self, *libraries):
        if not libraries:
            super().__init__(Constants())
        else:
            super().__init__(*libraries)

    def define_quantities(self):
        self.definitions = {'EOM_harmonic' : sy.diff(self.x, self.t, 2) + self.gamma * sy.diff(self.x, self.t) + self.Omega ** 2 * self.x - self.F,
                            'EOM_duffing': sy.diff(self.x, self.t, 2) + self.gamma * sy.diff(self.x, self.t) + self.Omega ** 2 * self.x - self.F + self.alpha * self.x ** 3,
                            'EOM_matthieu': sy.diff(self.x, self.t, 2) + self.gamma * sy.diff(self.x, self.t) + self.Omega ** 2 * (1 + self.lamda * sy.cos(2 * self.omega * self.t)) * self.x - self.F
                            }
    @classmethod
    def Potential(cls, force, variables):
        cls.curl(force, variables)

    def Potential(self, force, variables=None):
        if variables is None:
            variables = [self.x, self.y, self.z]
        curl = self.curl(force, variables)
        for c in curl:
            if c != 0:
                print('Curl appears to be non-zero. The derived Potential might be not reasonable.')
                sy.pprint(curl)
                break
        Phi_h_yz = sy.Function('\\hat{\\Phi}')(variables[1], variables[2])
        Phi_h_z = sy.Function('\\hat{\\Phi}')(variables[2])

        Phi_ = -1 * sy.integrate(force[0], variables[0]) + Phi_h_yz
        Phi_ = Phi_.subs(Phi_h_yz, sy.solve(sy.integrate(force[1] + sy.diff(Phi_, variables[1]), variables[1]), Phi_h_yz)[0] + Phi_h_z)
        Phi_ = sy.simplify(Phi_.subs(Phi_h_z, sy.solve(sy.integrate(force[2] + sy.diff(Phi_, variables[2]), variables[2]), Phi_h_z)[0]))
        return Phi_

    @staticmethod
    def curl(f, variables):
        if not isinstance(f, sy.Matrix):
            print('Function must be a Sympy Matrix with shape (3, 1).')
            return
        if len(variables) != 3:
            print('Three spatial variable for differentiation must be specified.')
            return
        return sy.Matrix([[sy.simplify(sy.diff(f[2], variables[1]) - sy.diff(f[1], variables[2]))],
                          [sy.simplify(sy.diff(f[0], variables[2]) - sy.diff(f[2], variables[0]))],
                          [sy.simplify(sy.diff(f[1], variables[0]) - sy.diff(f[0], variables[1]))]])

    @staticmethod
    def div(f, variables):
        if not isinstance(f, sy.Matrix):
            print('Function must be a Sympy Matrix with shape (3, 1).')
            return
        if len(variables) != 3:
            print('Three spatial variable for differentiation must be specified.')
            return
        return sy.simplify(sy.diff(f[0], variables[0]) + sy.diff(f[1], variables[1]) + sy.diff(f[2], variables[2]))

    @staticmethod
    def grad(f, variables):
        if not isinstance(f, sy.Expr):
            print('Function must be a Sympy Expression.')
            return
        if len(variables) != 3:
            print('Three spatial variable for differentiation must be specified.')
            return
        return sy.Matrix([[sy.simplify(sy.diff(f, variables[0]))],
                          [sy.simplify(sy.diff(f, variables[1]))],
                          [sy.simplify(sy.diff(f, variables[2]))]])


class Optics(Framework):
    symbols = {'E0' : [{'names' : 'E_0', 'real' : True}, 'Electric Field in Focus'],
               'z0' : [{'names' : 'z_0', 'positive' : True}, 'Rayleigh length'],
               'w0' : [{'names' : 'w_0', 'positive' : True}, 'Beam Waist'],
               'wx0': [{'names': 'w_x0', 'positive': True}, 'Beam Waist in x-direction'],
               'wy0': [{'names': 'w_y0', 'positive': True}, 'Beam Waist in y-direction'],
               'rho' : [{'names' : '\\rho', 'real' : True}, 'spatial zylindric coordinate'],
               'x' : [{'names' : 'x', 'real' : True}, 'spatial coordinate x'],
               'y' : [{'names' : 'y', 'real' : True}, 'spatial coordinate y'],
               'z' : [{'names' : 'z', 'real' : True}, 'spatial coordinate z'],
               'k' : [{'names' : 'k', 'positive' : True}, 'Wavenumber'],
               'E': [{'names': 'E'}, 'Electric Field'],
               'phi': [{'names': '\\phi', 'real': True}, 'Phase'],
               'w': [{'names': 'w', 'positive': True}, 'Beam Radius'],
               'wx': [{'names': 'w_x', 'positive': True}, 'Beam Width in x-direction'],
               'wy': [{'names': 'w_y', 'positive': True}, 'Beam Width in y-direction'],
               'R': [{'names': 'R', 'positive': True}, 'Wavefront Radius'],
               'HG00': [{'names': '\\mathrm{HG}_{00}'}, 'Hermitian-Gaussian 00 Mode'],
               'I_HG00': [{'names': 'I_{00}', 'positive' : True}, 'Intensity Hermitian-Gaussian 00 Mode'],
               'P_HG00': [{'names': 'P_{00}', 'positive': True}, 'Power Hermitian-Gaussian 00 Mode, Integrated Intensity'],
               'eta': [{'names': '\\eta', 'real': True}, 'Guoy Phase Shift'],
               'c': [{'names': 'c', 'positive': True}, 'Speed of Light'],
               'eps0': [{'names': '\\epsilon_0', 'positive': True}, 'Vacuum Permittivity'],
               'n': [{'names': 'n', 'positive': True}, 'Refractive Index'],
               'alpha_eff' : [{'names': '\\alpha_\\mathrm{eff}'}, 'Effective Polarizability'],
               'alpha': [{'names': '\\alpha'}, 'Polarizability'],
               'epsp': [{'names': '\\epsilon_p'}, 'Dielectric Constant Particle'],
               'epsm': [{'names': '\\epsilon_m'}, 'Dielectric Constant Medium'],
               'lamda': [{'names': '\\lambda'}, 'Wavelength'],
               'k_trap_x' : [{'names': 'k_\\mathrm{trap\,x}'}, 'Trap Stiffness in x direction'],
               'k_trap_y': [{'names': 'k_\\mathrm{trap\,y}'}, 'Trap Stiffness in y direction'],
               'k_trap_z': [{'names': 'k_\\mathrm{trap\,z}'}, 'Trap Stiffness in z direction'],
               'F_grad' : [{'names': 'F_\\mathrm{grad}'}, 'Gradient Force on a Dielectric Particle'],
               'F_grad_x': [{'names': 'F_\\mathrm{grad\,x}'}, 'x Component of Gradient Force on a Dielectric Particle'],
               'F_grad_y': [{'names': 'F_\\mathrm{grad\,y}'}, 'y Component of Gradient Force on a Dielectric Particle'],
               'F_grad_z': [{'names': 'F_\\mathrm{grad\,y}'}, 'z Component of Gradient Force on a Dielectric Particle'],
               'Phi_grad' : [{'names': '\\Phi_\\mathrm{grad}'}, 'Potential of Gradient Force on a Dielectric Particle'],
               'F_scat' : [{'names': 'F_\\mathrm{grad}'}, 'Scattering Force on a Dielectric Particle'],
               'F_scat_x': [{'names': 'F_\\mathrm{scat\,x}'}, 'x Component of Scattering Force on a Dielectric Particle'],
               'F_scat_y': [{'names': 'F_\\mathrm{scat\,y}'}, 'y Component of Scattering Force on a Dielectric Particle'],
               'F_scat_z': [{'names': 'F_\\mathrm{scat\,y}'}, 'z Component of Scattering Force on a Dielectric Particle'],
               'gamma0' : [{'names': '\\gamma_0'}, 'Scattering Field Constant'],
               'gammax': [{'names': '\\gamma_x'}, 'Scattering Field Constant'],
               'gammay': [{'names': '\\gamma_y'}, 'Scattering Field Constant'],
               'gammaz': [{'names': '\\gamma_z'}, 'Scattering Field Constant']
               }

    def __init__(self, *libraries):
        if not libraries:
            const = Constants()
            geom = Geometry(const)
            part = Particle(const, geom)
            super().__init__(const, geom, part)
        else:
            super().__init__(*libraries)

    def define_quantities(self):

        _F_grad_x_ = -1 * self.k_trap_x * (1 - 2 * self.x ** 2 / self.wx ** 2 - 2 * self.y ** 2 / self.wy ** 2 - 2 * self.z ** 2 / self.z0 ** 2) * self.x
        _F_grad_y_ = -1 * self.k_trap_y * (1 - 2 * self.x ** 2 / self.wx ** 2 - 2 * self.y ** 2 / self.wy ** 2 - 2 * self.z ** 2 / self.z0 ** 2) * self.y
        _F_grad_z_ = -1 * self.k_trap_z * (1 - 4 * self.x ** 2 / self.wx ** 2 - 4 * self.y ** 2 / self.wy ** 2 - 2 * self.z ** 2 / self.z0 ** 2) * self.z
        _F_scat_x_ = sy.im(self.alpha_eff) / sy.re(self.alpha_eff) * self.k_trap_z * (self.k * self.x * self.z)
        _F_scat_y_ = sy.im(self.alpha_eff) / sy.re(self.alpha_eff) * self.k_trap_z * (self.k * self.y * self.z)
        _F_scat_z_ = sy.im(self.alpha_eff) / sy.re(self.alpha_eff) * self.k_trap_z * (self.gamma0 + self.gammaz * self.z ** 2 + self.gammax * self.x ** 2 + self.gammay * self.y ** 2)
        _Phi_grad_ = self.k_trap_x * (1 - self.x ** 2 / self.wx ** 2) * self.x ** 2 / 2
        _Phi_grad_ += self.k_trap_y * (1 - self.y ** 2 / self.wy ** 2) * self.y ** 2 / 2
        _Phi_grad_ += self.k_trap_z * (1 - self.z ** 2 / self.z0 ** 2) * self.z ** 2 / 2
        _Phi_grad_ -= self.k_trap_y * self.x ** 2 * self.y ** 2 / self.wx ** 2
        _Phi_grad_ -= 2 * self.k_trap_z * self.x ** 2 * self.z ** 2 / self.wx ** 2
        _Phi_grad_ -= 2 * self.k_trap_z * self.y ** 2 * self.z ** 2 / self.wy ** 2

        self.definitions = {'phi' : self.k * self.z - self.eta + self.k * self.rho / (2 * self.R),
                            'w' : self.w0 * sy.sqrt(1 + self.z ** 2 / self.z0 ** 2),
                            'wx' : self.wx0 * sy.sqrt(1 + self.z ** 2 / self.z0 ** 2),
                            'wy': self.wy0 * sy.sqrt(1 + self.z ** 2 / self.z0 ** 2),
                            'R' : self.z * (1 + self.z0 ** 2 / self.z ** 2),
                            'HG00' : self.E0 * (1 + (self.z / self.z0) ** 2) ** (-1/2) * sy.exp(-(self.x ** 2 / self.wx ** 2 + self.y ** 2 / self.wy ** 2) + sy.I * self.phi),
                            'I_HG00' : self.c * self.n * self.eps0 / 2 * self.E0 ** 2 / (1 + (self.z / self.z0) ** 2) * sy.exp(-2 * (self.x ** 2 / self.wx ** 2 + self.y ** 2 / self.wy ** 2)),
                            'P_HG00' : self.c * self.n * self.eps0 / 4 * self.E0 ** 2 * sy.pi * self.wx0 * self.wy0,
                            'eta' : sy.atan2(self.z, self.z0),
                            'alpha': 3 * self.libraries['Geometry'].Vsp * self.eps0 * (self.epsp - self.epsm) / (self.epsp + 2 * self.epsm),
                            'alpha_eff': self.alpha * (1 - sy.I * self.k ** 3 / (6 * sy.pi * self.eps0) * self.alpha),
                            'k' : 2 * sy.pi / self.lamda,
                            'k_trap_x' : sy.re(self.alpha_eff) * self.E0 ** 2 / self.wx ** 2,
                            'k_trap_y': sy.re(self.alpha_eff) * self.E0 ** 2 / self.wy ** 2,
                            'k_trap_z': sy.re(self.alpha_eff) * self.E0 ** 2 / (2 * self.z0 ** 2),
                            'F_grad_x': _F_grad_x_,
                            'F_grad_y': _F_grad_y_,
                            'F_grad_z': _F_grad_z_,
                            'F_grad' : sy.Matrix([[_F_grad_x_], [_F_grad_y_],[_F_grad_z_]]),
                            'Phi_grad' : _Phi_grad_,
                            'F_scat_x': _F_scat_x_,
                            'F_scat_y': _F_scat_y_,
                            'F_scat_z': _F_scat_z_,
                            'F_scat' : sy.Matrix([[_F_scat_x_], [_F_scat_y_],[_F_scat_z_]]),
                            'gamma0' : self.z0 * (self.z0 * self.k - 1),
                            'gammaz' : (2 - self.z0 * self.k) / self.z0,
                            'gammax' : self.k / 2 - 2 * (self.z0 - self.k * self.z0 ** 2) / self.wx ** 2,
                            'gammay': self.k / 2 - 2 * (self.z0 - self.k * self.z0 ** 2) / self.wy ** 2
                            }

class Particle(Framework):
    symbols = {'gammag': [{'names': '\\gamma_\mathrm{gas}'}, 'Damping due to gas Collisions'],
               'Kn': [{'names': '\mathrm{Kn}'}, 'Knudsen Number'],
               'l': [{'names': 'l'}, 'Mean Free Path'],
               'm': [{'names': 'm'}, 'particle mass'],
               'r': [{'names': 'r'}, 'particle radius'],
               'c_k': [{'names': 'c_k'}, 'Knudsen flow'],
               'T': [{'names': 'T'}, 'Temperature'],
               'p_gas': [{'names': 'p_\mathrm{gas}'}, 'Gas Pressure']
               }

    def __init__(self, *libraries):
        if not libraries:
            const = Constants()
            super().__init__(const, Geometry(const))
        else:
            super().__init__(*libraries)

    def define_quantities(self):

        self.definitions = {'l' : self.libraries['Constants'].eta_Air / self.p_gas * sy.sqrt(sy.pi * self.libraries['Constants'].N_A * self.libraries['Constants'].k_B * self.T / (2 * self.libraries['Constants'].M_Air)),
                            'Kn' : self.l / self.r,
                            'gammag' : 6 * sy.pi * self.libraries['Constants'].eta_Air * self.r / self.m * 0.619 / (0.619 + self.Kn) * (1 + self.c_k),
                            'c_k' : 0.31 * self.Kn / (0.785 + 1.152 * self.Kn + self.Kn ** 2)}


class Constants(Framework):
    symbols = {'k_B': [{'names': 'k_\mathrm{B}'}, 'Boltzmann Constant'],
               'c': [{'names': 'c'}, 'Speed of Light'],
               'N_A': [{'names': 'N_\mathrm{A}'}, 'Avogadro Constant'],
               'M_Air': [{'names': 'M_\mathrm{Air}'}, 'Molar Mass Air'],
               'eta_Air': [{'names': '\\eta_\mathrm{Air}'}, 'Viscosity Coefficient Air']
               }

    def __init__(self, *libraries):
        if not libraries:
            super().__init__()
        else:
            super().__init__(*libraries)

    def define_quantities(self):

        self.definitions = {'k_B' : constants.k,
                            'c' : constants.c,
                            'N_A' : constants.N_A,
                            'M_Air' : 28.97e-3,          # kg/mol
                            'eta_Air' : 18.27e-6         # Pa * s
                            }


if __name__ == '__main__':
    o = Optics()
    print(o.libraries['Particle'].libraries)
    print(o.libraries)
