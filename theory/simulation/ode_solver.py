from numpy import zeros, arange


def ode_euler(func, y0, t, args=()):
    """Explicit Euler solver for integrating a system of ordinary
    differential equations.

    The solver works similarly to scipy.integrate.odeint,
    but additionally passes the time step dt to the evaluated
    function f(y, t, dt). The initial value problem is defined by

        dy/dt = func(y, t, dt, ...)

    The solver uses the explicit Euler method, where one time step
    is evaluated by

        y[i] = y[i-1] + dt * func(y[i-1], t[i-1], dt, ...)

    Parameters
    ----------
    func : callable(y, t, dt, ...)
        Computes the derivative of y at t.
    y0 : ndarray
        Initial condition on y (is a vector).
    t : ndarray
        A sequence of time points for which to solve for y.  The initial
        value point should be the first element of this sequence.
    args : tuple, optional
        Extra arguments to pass to function.

    Returns
    -------
    y : array, shape (len(t), len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value `y0` in the first row.
    """

    steps = len(t)

    y = zeros((steps, len(y0)))
    y[0] = y0

    for i in arange(steps - 1) + 1:
        dt = t[i] - t[i - 1]

        y[i] = y[i - 1] + dt * func(y[i - 1], t[i - 1], dt, *args)

    return y


def ode_runge_kutta(func, y0, t, args=()):
    """Runge Kutta solver for integrating a system of ordinary
    differential equations.

    The solver works similarly to scipy.integrate.odeint,
    but additionally passes the time step dt to the evaluated
    function f(y, t, dt). The initial value problem is defined by

        dy/dt = func(y, t, dt, ...)

    The solver uses the classical Runge Kutta method (RK4), where one
    time step is evaluated by

        y[i] = y[i-1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        k1 = func(y[i-1], t[i-1], dt, ...)
        k2 = func(y[i-1] + dt / 2 * k1, t[i-1] + dt / 2, dt, ...)
        k3 = func(y[i-1] + dt / 2 * k2, t[i-1] + dt / 2, dt, ...)
        k4 = func(y[i-1] + dt * k3, t[i-1] + dt, dt, ...)

    Parameters
    ----------
    func : callable(y, t, dt, ...)
        Computes the derivative of y at t.
    y0 : ndarray
        Initial condition on y (is a vector).
    t : ndarray
        A sequence of time points for which to solve for y.  The initial
        value point should be the first element of this sequence.
    args : tuple, optional
        Extra arguments to pass to function.

    Returns
    -------
    y : array, shape (len(t), len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value `y0` in the first row.

    Examples
    --------
    In this example we integrate the EOM of a particle in a 3D optical
    trap, subject to gas damping. We have to first define the initial
    value problem, initial values y0 and a time vector t. Afterwards
    we can call ode_runge_kutta to integrate the differential equation.

    >>> from numpy import array, pi
    >>> from optomechanics.theory.optical_force import \
    >>>     harmonic_force
    >>> from optomechanics.theory.damping_heating import \
    >>>     fluctuating_force
    >>>
    >>> gamma = 1000*2*pi
    >>> radius = 136e-9/2
    >>> m=2200*4/3*pi*radius**3
    >>> T = 300
    >>>
    >>> def initial_value_problem(position_velocity_vector, t, dt):
    >>>     x, y, z, vx, vy, vz = position_velocity_vector
    >>>
    >>>     Fopt_x, Fopt_y, Fopt_z = harmonic_force(
    >>>         array([x, y, z]), array([125, 140, 40])*1e3*2*pi, m)
    >>>     Ffl_x, Ffl_y, Ffl_z = fluctuating_force(
    >>>         gamma, m, T, dt, (3,))
    >>>
    >>>     f_vx = -gamma*vx + (Fopt_x + Ffl_x) / m
    >>>     f_vy = -gamma*vy + (Fopt_y + Ffl_y) / m
    >>>     f_vz = -gamma*vz + (Fopt_z + Ffl_z) / m
    >>>
    >>>     f_x = vx
    >>>     f_y = vy
    >>>     f_z = vz
    >>>
    >>>     return array([f_x, f_y, f_z, f_vx, f_vy, f_vz])
    >>>
    >>> y0 = [0, 0, 1e-20, 0, 0, 0]
    >>> t = arange(0, 1e-2, 1e-7)
    >>>
    >>> position_velocity = ode_runge_kutta(initial_value_problem,
    >>>                                     y0, t)
    """

    steps = len(t)

    y = zeros((steps, len(y0)))
    y[0] = y0

    for i in arange(steps - 1) + 1:
        dt = t[i] - t[i - 1]

        k1 = func(y[i - 1], t[i - 1], dt / 4, *args)
        k2 = func(y[i - 1] + dt * k1 / 2, t[i - 1] + dt / 2, dt / 4,
                  *args)
        k3 = func(y[i - 1] + dt * k2 / 2, t[i - 1] + dt / 2, dt / 4,
                  *args)
        k4 = func(y[i - 1] + dt * k3, t[i - 1] + dt, dt / 4, *args)

        y[i] = y[i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y
