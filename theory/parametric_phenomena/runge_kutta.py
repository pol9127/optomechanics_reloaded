import numpy as np
from scipy.integrate import complex_ode, quad
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
from scipy.special import mathieu_a, mathieu_b, mathieu_cem, mathieu_sem
import peakutils
import sympy as sy

class OscillatorSystem:
    '''
    This class provides tools to analyse differential equations of the form x''(t) + g1(t)x'(t) + g0(t)x(t) + alpha * x ** 3 = f(t)
    '''
    _g_1 = 0
    _g_0 = 4 * np.pi ** 2
    _f = 0
    _alpha = 0
    x_0 = np.array([1, 0])
    t_0 = 0
    initial_value_problem = None
    wronskian_matrix = None
    wronskian = None
    state_transition_matrix = None
    characteristic_exponent = None
    transformation = False

    t = sy.symbols('t', real=True)
    omega, omega0, lamda, alpha, Gamma = sy.symbols('\\omega \\omega_0 \\lambda \\alpha \\Gamma', positive=True)
    x = sy.Function('x')(t)
    p = sy.Function('p')(t)
    xprime = sy.Function('x\'')(t)
    pprime = sy.Function('p\'')(t)
    u = sy.Function('u')(t)
    v = sy.Function('v')(t)
    F = sy.Function('F')(x, p, t)
    U, V, Uprime, Vprime = sy.symbols('U V U\' V\'', real=True)
    theta, eta, F0 = sy.symbols('\\theta \\eta F_0', real=True)

    slow_flow_equations = {Uprime : -sy.pi*(U ** 3 * eta * omega + 3 * U ** 2 * V * alpha + U * V ** 2 * eta * omega + 4 * U
                                            * Gamma * omega + 2 * U * lamda * omega0 ** 2 * sy.sin(theta) + 3 * V ** 3
                                            * alpha - 2 * V * lamda * omega0 ** 2 * sy.cos(theta) - 4 * V * omega ** 2
                                            + 4 * V * omega0 ** 2) / (4 * omega ** 2),
                           Vprime : -sy.pi * ( 4 * F0 - 3 * U ** 3 * alpha + U ** 2 * V * eta * omega - 3 * U
                                               * V ** 2 * alpha - 2 * U * lamda * omega0 ** 2 * sy.cos(theta) + 4 * U
                                               * omega ** 2 - 4 * U * omega0 ** 2 + V ** 3 * eta * omega + 4 * V * Gamma
                                               * omega - 2 * V * lamda * omega0 ** 2 * sy.sin(theta)) / (4 * omega ** 2)}
    _slow_flow_equations_np = {Uprime : sy.lambdify([U, V, omega, omega0, Gamma, lamda, alpha, eta, theta, F0],
                                                    slow_flow_equations[Uprime], 'numpy'),
                               Vprime : sy.lambdify([U, V, omega, omega0, Gamma, lamda, alpha, eta, theta, F0],
                                                    slow_flow_equations[Vprime], 'numpy')}
    slow_flow_hessian_det = sy.det(sy.Matrix([[sy.diff(slow_flow_equations[Uprime], U), sy.diff(slow_flow_equations[Uprime], V)],
                                              [sy.diff(slow_flow_equations[Vprime], U), sy.diff(slow_flow_equations[Vprime], V)]]))
    _slow_flow_hessian_det_np = [sy.lambdify([U, V, omega, omega0, Gamma, lamda, alpha, eta, theta, F0],
                                              slow_flow_hessian_det, 'numpy')]
    @property
    def g_0(self):
        return self._g_0

    @g_0.setter
    def g_0(self, g_0_new):
        self._g_0 = g_0_new

    @property
    def g_1(self):
        return self._g_1

    @g_1.setter
    def g_1(self, g_1_new):
        self._g_1 = g_1_new

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f_new):
        self._f = f_new

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_new):
        self._alpha = alpha_new

    def equation_of_motion(self, t, x):
        if callable(self.g_0):
            g_0 = self.g_0(t)
        else:
            g_0 = self.g_0
        if callable(self.g_1):
            g_1 = self.g_1(t)
        else:
            g_1 = self.g_1
        if callable(self.f):
            f = self.f(t)
        else:
            f = self.f
        if callable(self.alpha):
            alpha = self.alpha(t)
        else:
            alpha = self.alpha

        return np.array([x[1], -g_0 * x[0] - g_1 * x[1] - alpha * x[0] ** 3 + f])
        # return ((np.array([[0, 1],
        #                    [-g_0, -g_1]]) @ x).T + np.array([0, f])).T

    # def initialize_mathieu_eqn(self, a, q):
    #     self.g_1 = 0
    #     self.alpha = 0
    #     self.g_0 = lambda t: a ** 2 - 2 * q * np.cos(2 * t)
    #     self.wronskian_matrix = lambda t: np.array([mathieu_cem(a, q, t * 360 / (2 * np.pi)),
    #                                                 mathieu_sem(a, q, t * 360 / (2 * np.pi))]).T
    #     def _state_transition_matrix(t, tau):
    #         wronskian_mat = self.wronskian_matrix(t)
    #         wronskian_mat_inv = self.wronskian_matrix(tau)
    #         return wronskian_mat @ np.array([[wronskian_mat_inv[1, 1], -1 *wronskian_mat_inv[0, 1]],
    #                                          [-1 * wronskian_mat_inv[1, 0], wronskian_mat_inv[0, 0]]]) / (
    #             wronskian_mat_inv[0, 0] * wronskian_mat_inv[1, 1] - wronskian_mat_inv[0, 1] * wronskian_mat_inv[1, 0])
    #
    #     self.state_transition_matrix = _state_transition_matrix
    #     self.characteristic_exponent = np.array([mathieu_a(a, q), mathieu_b(a, q)])
    #     self.transformation = False

    def initialize_undamped_harmonic(self, omega_0, driving=0):
        self.g_1 = 0
        self.alpha = 0
        self.g_0 = omega_0 ** 2
        self.f = driving
        self.wronskian_matrix = lambda t: np.array([[np.sin(omega_0 * t), np.cos(omega_0 * t)],
                                                    [omega_0 * np.cos(omega_0 * t), -omega_0 * np.sin(omega_0 * t)]])
        def _state_transition_matrix(t, tau):
            wronskian_mat = self.wronskian_matrix(t)
            wronskian_mat_inv = self.wronskian_matrix(tau)
            return wronskian_mat @ np.array([[wronskian_mat_inv[1, 1], -1 * wronskian_mat_inv[0, 1]],
                                             [-1 * wronskian_mat_inv[1, 0], wronskian_mat_inv[0, 0]]]) / (
                wronskian_mat_inv[0, 0] * wronskian_mat_inv[1, 1] - wronskian_mat_inv[0, 1] * wronskian_mat_inv[1, 0])

        self.state_transition_matrix = _state_transition_matrix

        T = 2 * np.pi / omega_0
        char_exp = np.arccosh(np.trace(self.state_transition_matrix(T, 0)) / 2) / T
        self.characteristic_exponent = np.array([char_exp, -1 * char_exp])
        self.transformation = False

    def initialize_damped_harmonic(self, omega_0, gamma, driving=0):
        self.g_1 = gamma
        self.alpha = 0
        self.g_0 = omega_0 ** 2
        self.f = driving
        omega_0 = np.sqrt(self.g_0 - 0.25 * self.g_1 ** 2)
        self.wronskian_matrix = lambda t: np.array([[np.sin(omega_0 * t), np.cos(omega_0 * t)],
                                                    [omega_0 * np.cos(omega_0 * t), -omega_0 * np.sin(omega_0 * t)]])
        def _state_transition_matrix(t, tau):
            wronskian_mat = self.wronskian_matrix(t)
            wronskian_mat_inv = self.wronskian_matrix(tau)
            return wronskian_mat @ np.array([[wronskian_mat_inv[1, 1], -1 * wronskian_mat_inv[0, 1]],
                                             [-1 * wronskian_mat_inv[1, 0], wronskian_mat_inv[0, 0]]]) / (
                wronskian_mat_inv[0, 0] * wronskian_mat_inv[1, 1] - wronskian_mat_inv[0, 1] * wronskian_mat_inv[1, 0])

        self.state_transition_matrix = _state_transition_matrix

        T = 2 * np.pi / omega_0
        char_exp = np.arccosh(np.trace(self.state_transition_matrix(T, 0)) / 2) / T
        self.characteristic_exponent = np.array([char_exp, -1 * char_exp])
        self.transformation = True

    def state_transition_evolution(self, t):
        if self.state_transition_matrix is None:
            print('ERROR: No State Transition Matrix defined. Abort.')
            return
        if self.transformation:
            if callable(self.g_1):
                x_0 = np.array([self.x_0[0], (self.x_0[1] + 0.5 * self.g_1(0) * self.x_0[0])])
            else:
                x_0 = np.array([self.x_0[0], (self.x_0[1] + 0.5 * self.g_1 * self.x_0[0])])
        else:
            x_0 = self.x_0

        if self.f == 0:
            if not self.transformation:
                return np.array([self.state_transition_matrix(t_tmp, 0) @ x_0 for t_tmp in t])
            else:
                if callable(self.g_1):
                    return np.array([(np.exp(-0.5 * quad(self.g_1, 0, t_tmp)) * np.array([[1 , 0],
                                                                                          [-0.5 * self.g_1(t_tmp), 1]])) @
                                     (self.state_transition_matrix(t_tmp, 0) @ x_0) for t_tmp in t])
                else:
                    return np.array([(np.exp(-0.5 * self.g_1 * t_tmp) * np.array([[1 , 0],
                                                                                  [-0.5 * self.g_1, 1]])) @
                                     (self.state_transition_matrix(t_tmp, 0) @ x_0) for t_tmp in t])
        else:
            if self.transformation:
                if callable(self.f):
                    if callable(self.g_1):
                        f = lambda t: self.f(t) / np.exp(-0.5 * quad(self.g_1, 0, t))
                    else:
                        f = lambda t: self.f(t) / np.exp(-0.5 * self.g_1 * t)
                else:
                    if callable(self.g_1):
                        f = lambda t: self.f / np.exp(-0.5 * quad(self.g_1, 0, t))
                    else:
                        f = lambda t: self.f / np.exp(-0.5 * self.g_1 * t)

            if callable(f):
                driving_evo_0 = lambda tau, t: (self.state_transition_matrix(t, tau) @ np.array([0, f(tau)]))[0]
                driving_evo_1 = lambda tau, t: (self.state_transition_matrix(t, tau) @ np.array([0, f(tau)]))[1]
            else:
                driving_evo_0 = lambda tau, t: self.state_transition_matrix(t, tau)[0, 1] * f
                driving_evo_1 = lambda tau, t: self.state_transition_matrix(t, tau)[1, 1] * f

            if not self.transformation:
                return np.array([self.state_transition_matrix(t_tmp, 0) @ x_0 +
                                 np.array([quad(driving_evo_0, 0, t_tmp, args=(t_tmp))[0],
                                           quad(driving_evo_1, 0, t_tmp, args=(t_tmp))[0]]) for t_tmp in t])

            else:
                if callable(self.g_1):
                    return np.array([(np.exp(-0.5 * quad(self.g_1, 0, t_tmp)) * np.array([[1 , 0],
                                                                                          [-0.5 * self.g_1(t_tmp), 1]])) @
                                     (self.state_transition_matrix(t_tmp, 0) @ x_0 + np.array([quad(driving_evo_0, 0, t_tmp, args=(t_tmp))[0],
                                                                                                    quad(driving_evo_1, 0, t_tmp, args=(t_tmp))[0]])) for t_tmp in t])
                else:
                    return np.array([(np.exp(-0.5 * self.g_1 * t_tmp) * np.array([[1 , 0],
                                                                                  [-0.5 * self.g_1, 1]])) @
                                     (self.state_transition_matrix(t_tmp, 0) @ x_0 + np.array([quad(driving_evo_0, 0, t_tmp, args=(t_tmp))[0],
                                                                                                    quad(driving_evo_1, 0, t_tmp, args=(t_tmp))[0]])) for t_tmp in t])

    def solve_eqn(self, t):
        dt = t[1] - t[0]
        self.initial_value_problem = complex_ode(self.equation_of_motion)
        self.initial_value_problem.set_initial_value(self.x_0, self.t_0)
        x = [self.x_0]
        for t_tmp in t[:-1]:
            self.initial_value_problem.integrate(self.initial_value_problem.t + dt)
            x.append(self.initial_value_problem.y)
        return np.array(x)

    def fft(self, t, x):
        return np.fft.rfftfreq(len(x), t[1] - t[0]), np.abs(np.fft.rfft(x))

    def slow_flow_gradient(self, uv, slow_flow_params):
        return np.vstack((self._slow_flow_equations_np[self.Uprime](uv[0], uv[1], *slow_flow_params),
                          self._slow_flow_equations_np[self.Vprime](uv[0], uv[1], *slow_flow_params)))

    def slow_flow_hessian_det_np(self, uv, slow_flow_params):
        return self._slow_flow_hessian_det_np[0](uv[0], uv[1], *slow_flow_params)


    def streamplot(self, x, p, slow_flow_params=None, fig_nr=None):
        X, P = np.meshgrid(x, p)
        XP = np.vstack((X.flatten(), P.flatten()))
        if fig_nr is None:
            fig, ax = plt.subplots()
        else:
            plt.figure(fig_nr)
            ax = plt.gca()
        if slow_flow_params is None:
            UV = self.equation_of_motion(0, XP)
        else:
            UV = self.slow_flow_gradient(XP, slow_flow_params)
        U = UV[0].reshape(X.shape)
        V = UV[1].reshape(P.shape)
        ax.streamplot(X, P, U, V)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(p[0], p[-1])
        plt.show()
        return plt.gcf().number

    @property
    def duffing_frequency(self):
        if callable(self.g_0) or callable(self.alpha):
            print('Frequency and duffing coefficient must be constant in time.')
        else:
            return np.sqrt(self.g_0) *( 1 + 3/8 * self.x_0[0] ** 2 * self.alpha / np.sqrt(self.g_0) ** 2 - 21 * self.x_0[0] ** 4 / 256 * self.alpha ** 2 / np.sqrt(self.g_0) ** 4)

    def kinetic_energy(self, x, p, t=None):
        if callable(self.g_1) and t is None:
            print('Damping must be constant or time must be specified.')
        else:
            if callable(self.g_1):
                g_1 = self.g_1(t)
            else:
                g_1 = self.g_1
            return 0.5 * (p + g_1 * x) ** 2

    def potential_energy(self, x, p, t=None):
        if callable(self.g_0) or callable(self.alpha) and t is None:
            print('Frequency and duffing coefficient must be constant in time or time must be specified.')
        else:
            if callable(self.g_0):
                g_0 = self.g_0(t)
            else:
                g_0 = self.g_0
            if callable(self.alpha):
                alpha = self.alpha(t)
            else:
                alpha = self.alpha
            return 0.5 * g_0 * x ** 2 + 0.25 * alpha * x ** 4

    def envelope(self, data, time):
        indexes_max = peakutils.indexes(data, thres=0.00001, min_dist=1)
        indexes_min = peakutils.indexes(-1 * data, thres=0.00001, min_dist=1)
        return [np.vstack((time[indexes_max], data[indexes_max])), np.vstack((time[indexes_min], data[indexes_min]))]


if __name__ == '__main__':
    os = OscillatorSystem()


    os.g_0 = 1
    os.g_1 = 0.05
    os.alpha = 0.001
    T = 130000
    f1 = 1.2
    f0 = 0.9
    F = -1
    os.f = lambda t: F * np.cos(((f1 - f0)*t/T + f0) * t)
    os.x_0 = np.array([0.1, 0.])

    # fig, ax = plt.subplots()
    # x = np.linspace(-10, 10, 100)
    # ax.plot(x, os.potential_energy(x, 0))
    # os.alpha = 0.
    # ax.plot(x, os.potential_energy(x, 0))
    # plt.show()

    # print(np.sqrt(os.g_0 / os.g_1 - 0.25))
    # os.initialize_mathieu_eqn(10, 0)
    # os.initialize_undamped_harmonic(10, 1)
    # os.initialize_damped_harmonic(10, 1)

    # x = np.linspace(-10, 10, 100)
    # p = np.linspace(-10, 10, 100)
    # os.streamplot(x, p)

    t = np.linspace(0, T, 500000)
    fig, ax = plt.subplots(1, 2)
    x = os.solve_eqn(t)
    # x1 = os.state_transition_evolution(t)

    ax[0].plot(((f1 - f0)*t/T + f0), x[:, 0])
    # ax[0].plot(t, x1[:, 0])
    ax[1].semilogy(*os.fft(t, x[:, 0]))
    # ax[1].semilogy(*os.fft(t, x1[:, 0]))

    ax[0].grid(True)
    ax[1].grid(True)

    # os.x_0 = x[-1, :]
    # os.x_0[1] *= -1
    # os.f = lambda t: F * np.cos(((f1 - f0)*(T- t)/T + f0) * (T- t))
    #
    # # fig, ax = plt.subplots(1, 2)
    # x = os.solve_eqn(t)
    # # x1 = os.state_transition_evolution(t)
    #
    # ax[0].plot(((f1 - f0)*(T- t)/T + f0), x[:, 0], alpha = 0.3)
    # # ax[0].plot(t, x1[:, 0])
    # ax[1].semilogy(*os.fft(t, x[:, 0]))
    # # ax[1].semilogy(*os.fft(t, x1[:, 0]))
    #
    # ax[0].grid(True)
    # ax[1].grid(True)


    plt.show()

    #############################

