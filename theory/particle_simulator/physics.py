import json
import numpy as np
import sympy as sy
from collections.abc import Iterable
import dill
import os
from scipy import constants as con


class Equations:
    parameters = None
    numeric_values = None
    parameter_mapping = None
    equations = None
    formulas = None
    parameter_descriptor = None

    def __init__(self, numeric_values=None):
        self.numeric_values = numeric_values
        self.load_parameters()
        self.parameter_mapping = {}
        self.parameter_descriptor = {}

        for key in self.parameters:
            exec('global '+ key + '; ' + key + '= sy.Symbol(self.parameters[key]["tooltip"], **self.parameters[key]["kwargs"])')
            exec('self.parameter_mapping[' + key + '] = "' + key + '"')
            exec('self.parameter_descriptor["' + key + '"] = self.parameters[key]["descrp"]')

        # Basics
        lamda_ = 2 * sy.pi / k
        omega_ = c_0 * k
        z_R_ = sy.pi * w_x0 * w_y0 / lamda
        V_ = 4 / 3 * sy.pi * r_x * r_y * r_z
        rho_ = m / V

        # Hebestreit Thesis eqn 2.(4 - 26)
        alpha_p_ = 3 * eps_0 * V * (eps_p - eps_m) / (eps_p + 2 * eps_m)
        E_0_ = sy.sqrt(4 * P / (c_0 * eps_0 * sy.pi * w_x0 * w_y0))
        k_x_ = alpha_p * E_0 ** 2 / w_x0 ** 2
        k_y_ = alpha_p * E_0 ** 2 / w_y0 ** 2
        k_z_ = alpha_p * E_0 ** 2 / z_R ** 2
        xi_x_ = -2 / w_x0 ** 2
        xi_y_ = -2 / w_y0 ** 2
        xi_z_ = -2 / z_R ** 2

        gamma_x_gas_ = 6 * sy.pi * eta * r_x / m * 0.619 / (0.619 + Kn_x) * (1 + c_xk)
        gamma_y_gas_ = 6 * sy.pi * eta * r_y / m * 0.619 / (0.619 + Kn_y) * (1 + c_yk)
        gamma_z_gas_ = 6 * sy.pi * eta * r_z / m * 0.619 / (0.619 + Kn_z) * (1 + c_zk)

        c_xk_ = 0.31 * Kn_x / (0.785 + 1.152 * Kn_x + Kn_x ** 2)
        c_yk_ = 0.31 * Kn_y / (0.785 + 1.152 * Kn_y + Kn_y ** 2)
        c_zk_ = 0.31 * Kn_z / (0.785 + 1.152 * Kn_z + Kn_z ** 2)

        Kn_x_ = l / r_x
        Kn_y_ = l / r_y
        Kn_z_ = l / r_z

        l_ = eta / p_gas * sy.sqrt(sy.pi * N_A * k_B * T / (2 * M))

        Omega_x0_ = sy.sqrt(k_x / m)
        Omega_y0_ = sy.sqrt(k_y / m)
        Omega_z0_ = sy.sqrt(k_z / m)

        self.equations = {
            lamda: lamda_,
            omega: omega_,
            rho: rho_,
            z_R: z_R_,
            alpha_p: alpha_p_,
            E_0: E_0_,
            k_x: k_x_,
            k_y: k_y_,
            k_z: k_z_,
            xi_x: xi_x_,
            xi_y: xi_y_,
            xi_z: xi_z_,
            V: V_,
            l: l_,
            gamma_x_gas: gamma_x_gas_,
            gamma_y_gas: gamma_y_gas_,
            gamma_z_gas: gamma_z_gas_,
            c_xk: c_xk_,
            c_yk: c_yk_,
            c_zk: c_zk_,
            Kn_x: Kn_x_,
            Kn_y: Kn_y_,
            Kn_z: Kn_z_,
            Omega_x0: Omega_x0_,
            Omega_y0: Omega_y0_,
            Omega_z0: Omega_z0_
                    }

        self.load_formulas()

    def load_parameters(self):
        current_location = os.path.split(os.path.realpath(__file__))[0]
        with open(os.path.join(current_location, 'parameters.json')) as parameters_file:
            self.parameters = json.load(parameters_file)

    def derive_and_export_formulas(self):
        free_symbols = []
        for key in self.equations:
            free_tmp = list(self.equations[key].free_symbols)
            free_tmp = [element for element in free_tmp if element not in list(self.equations.keys()) + free_symbols]
            free_symbols += free_tmp

        solutions = self.equations.copy()
        for key in self.equations:
            eq = self.equations[key] - key
            sols = {symbol: sy.solve(eq, symbol) for symbol in list(eq.free_symbols) if symbol not in self.equations}
            for symbol in sols:
                if symbol not in solutions:
                    solutions[symbol] = sols[symbol]
        else:
            solutions[symbol] += sols[symbol]

        solutions_np = {}
        for key in solutions:
            sols = solutions[key]
            if not isinstance(sols, Iterable):
                sols = [sols]
            sols_np = []
            for sol in sols:
                params = list(sol.free_symbols)
                params_labels = [self.parameter_mapping[param] for param in params]
                function = sy.lambdify(params, sol, 'numpy')
                sols_np.append((params_labels, function))
            solutions_np[self.parameter_mapping[key]] = sols_np

        current_location = os.path.split(os.path.realpath(__file__))[0]
        with open(os.path.join(current_location, 'solutions_np.pkl'), 'wb') as solutions_np_file:
            dill.dump(solutions_np, solutions_np_file, recurse=True)

    def load_formulas(self):
        current_location = os.path.split(os.path.realpath(__file__))[0]
        if not os.path.exists(os.path.join(current_location, 'solutions_np.pkl')):
            self.derive_and_export_formulas()

        with open(os.path.join(current_location, 'solutions_np.pkl'), 'rb') as solutions_np_file:
            self.formulas = dill.load(solutions_np_file)

    def dump_parameters(self):
        current_location = os.path.split(os.path.realpath(__file__))[0]
        with open(os.path.join(current_location, 'parameters.json'), 'w') as parameters_file:
            json.dump(self.parameters, parameters_file, indent=1)

    def find_numeric_values(self):
        if self.numeric_values is None:
            return
        n = 1
        while n > 0:
            n = 0
            for key in self.formulas:
                if key in self.numeric_values:
                    continue
                sols = self.formulas[key]
                for sol in sols:
                    if all([param in self.numeric_values for param in sol[0]]):
                        param_values = [self.numeric_values[param] for param in sol[0]]
                        self.numeric_values[key] = sol[1](*param_values)
                        n += 1
                        break


if __name__ == '__main__':
    numeric_values = {
        'c_0': con.c,
        'eps_0': con.epsilon_0,
        'N_A': con.N_A,
        'k_B': con.k,
        'eps_m': 1,         # in vacuum
        'eps_p': 1.44 ** 2, # at 1550e-9 nm wavelength
        'M': 28.97e-3,      # kg/mol (Hebestreit thesis, p14)
        'eta': 18.27e-6,    # Pa * s (Hebestreit thesis, p14)
        'T': 300,
        'p_gas': 1e5,
        'w_x0': 1e-6,
        'w_y0': 1e-6,
        'lamda': 1550e-9,
        'P': 400e-3,
        'r_x': 70e-9,
        'r_y': 70e-9,
        'r_z': 70e-9,
        'rho': 2200         # density of Silica
    }

    equations = Equations(numeric_values)
    equations.find_numeric_values()
