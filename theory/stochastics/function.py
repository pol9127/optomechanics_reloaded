"""
@ author: Andrei Militaru
@ Date: 14th of December 2018

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ...visualization.set_axes import set_ax
from numba import njit

from tqdm import tqdm


def foo():
    return None


@njit()
def fact(n):
    output = 1
    for i in range(n):
        output *= (i + 1)
    return output


@njit()
def put_in_grid(grid,
                element,
                mode):
    """
    Given a grid and an element, it returns the place of the element
    the grid.
    -----------------
    Parameters:
        grid: indexed array of ordered floats
            the grid in which the element must be placed
        element: float
            the element that needs to be placed in the grid
        mode: str, optional
            options are 'binary', for a binary search in the grid,
                        'exhaustive' for a search element by element.
            Defaults to 'binary'
    -----------------
    Returns:
        int:
            the index corresponding to the place of element into grid.
    -----------------
    """

    if mode == 'exhaustive':
        found_place = False
        for i in range(len(grid) - 1):
            if grid[i] <= element < grid[i + 1]:
                position = i
                found_place = True
            else:
                continue
        return position if found_place else None

    elif mode == 'binary':
        if element > grid[-1] or element < grid[0]:
            return None
        else:
            step = grid[1] - grid[0]
        return int((element - grid[0]) // step)

@njit()
def get_D_nb(realization,
             grid,
             df):
    """
    Numba optimized computation for the drift and diffusion estimation.
    ---------------
    Parameters:
         realization: numpy.ndarray
            Single realization of the stochastic diffusion of interest.
         grid: numpy.ndarray
            Equally spaced spatial grid on which drift and diffusion need to
            be evaluated.
         df: float
            Sampling frequency of the realization.
    """

    L = len(grid)
    grid_step = grid[1] - grid[0]
    drifts = np.zeros(L)
    diffs = np.zeros(L)
    counts = np.zeros(L)

    for i in range(len(realization) - 1):
        position = realization[i]
        if grid[0] < position < grid[-1]:
            grid_index = int((position - grid[0]) // grid_step)
            drift = (realization[i+1] - position) * df
            diff = (realization[i+1] - position)**2 * df / 2
            counts[grid_index] += 1
            drifts[grid_index] += drift

            diffs[grid_index] += diff
        else:
            pass

    for i in range(L):
        if counts[i] != 0:
            drifts[i] /= counts[i]
            diffs[i] /= counts[i]
        else:
            pass

    return (grid[counts != 0], drifts[counts != 0], diffs[counts != 0])


@njit()
def get_higher_D_nb(n,
                    realization,
                    grid,
                    df):
    """

    Numba optimized computation for the computation of Dn coefficients (n > 2).
    The coefficients are defined as \lim_{\Delta t \to 0} E[(z(t+\Delta t) - z(t))^n/(n!\Delta t)],
    where z(t) is the realization.
    ---------------
    Parameters:
        n: int,
            order of the D coefficient.
        realization: numpy.ndarray
            Single realization of the stochastic diffusion of interest.
        grid: numpy.ndarray
            Equally spaced spatial grid on which drift and diffusion need to
            be evaluated.
        df: float
            Sampling frequency of the realization.
    """

    L = len(grid)
    grid_step = grid[1] - grid[0]
    Ds = np.zeros(L)
    counts = np.zeros(L)

    for i in range(len(realization) - 1):
        position = realization[i]
        if grid[0] < position < grid[-1]:
            grid_index = int((position - grid[0]) // grid_step)
            D = (realization[i + 1] - position) ** n * df / fact(n)
            counts[grid_index] += 1
            Ds[grid_index] += D
        else:
            pass

    for i in range(L):
        if counts[i] != 0:
            Ds[i] /= counts[i]
        else:
            pass

    return (grid[counts != 0], Ds[counts != 0])

class function():
    
    def __init__(self,
                 func,
                 var = 1):
        """
        -------------------
        Parameters:
            func: function
                when calculating the returned value of this object, the function 
                func will be used. func needs to be defined somewhere in the file that makes
                use of this class.
            var: int
                number of arguments that func must receive. So far, not used.
        -------------------
        """
        
        self.function = func
        self.single_variable = (var == 1)   # no method makes currently use of this
        self.dimensions = var               # no method makes currently use of this
        self.name = func.__name__
        
    """
    -------------------------
    Overloading the most common operations, such that they can be used between
    instantiations of this class.
    -------------------------
    """
    
    def __add__(self,other):
    # overload the addition
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
            
        def overloaded(*args):
            return (self.eval(*args) + other.eval(*args))
        return function(overloaded)

    def __sub__(self,other):
    #overload the subtraction
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
        def overloaded(*args):
            return (self.eval(*args) - other.eval(*args))
        return function(overloaded)

    def __mul__(self,other):
    # overload the multiplication
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
            
        def overloaded(*args):
            return (self.eval(*args) * other.eval(*args))
        return function(overloaded)

    def __rmul__(self,other):
    # overload the multiplication where an instantiation comes second
        return self.__mul__(other)

    def __rtruediv__(self,other):
    # overload the division when an instantiation comes second
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
        return other.__truediv__(self)

    def __radd__(self,other):
    # overload the addition when an instantiation comes second
        return self.__add__(other)

    def __rsub__(self,other):
    # overload the addition when an instantiation comes second
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
        return other.__sub__(self)

    def __pow__(self,other):
    # overload the power operation
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
            
        def overloaded(*args):
            return (self.eval(*args)**other.eval(*args))
        return function(overloaded)

    def __truediv__(self,other):
    # overload the division
        if type(other) is int or type(other) is float:
            temp = other
            def objectivize(*args):
                return temp
            other = function(objectivize)
            
        def overloaded(*args):
            return (self.eval(*args)/other.eval(*args))
        return function(overloaded)

    def __abs__(self):
    # overload the abs() function 
        def overloaded(*args):
            return abs(self.eval(*args))
        return function(overloaded)
        
    """
    -----------------------
    End of the overloading section of the class.
    -----------------------
    """

    def eval(self,
             *args):
        """
        -----------------
        Parameters:
            args: tuple of floats
                parameters given as input to self.function
        ----------------
        Returns:
            returned value by self.function
        -----------------
        """
             
        return self.function(*args)

    def pder(self,          
             *args,  
             variable = 0, 
             h = 1e-6,  
             mode = 'five-point'): 
        """
        This method performs the partial derivative of a function evaluated in the coordinates args.
        If the function is one-dimensional it automatically calculates the total derivative.
        ------------------------------
        Parameters:
            args: tuple of floats
                passed as parameters to self.function
            variable: int or str, optional
                the variable with respect to which the numeric derivative needs to be calculated.
                'x','y' and 'z' are respectively interpreted as 0,1,2.
                Defaults to 0.
            h: float, optional
                finite step used for the incremental ratio. Defaults to 1e-6.
            mode: string, optional
                numerical method used for the evaluation of the derivative. Available options:
                    - 'increment': f'(x) = 1/h * (f(x+h)-f(x))
                        order of error: h
                    - 'decrement': f'(x) = 1/h * (f(x)-f(x-h))
                        order of error: h
                    - 'symmetric': f'(x) = 1/(2*h) * (f(x+h)-f(x-h))
                        order of error: h**2  
                    - 'five-point': f'(x) = 1/(12*h) * (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))
                        order of error: h**4
                        
                Defaults to 'five-point', which is the most precise method.
        ------------------------------
        Returns:
            type of output is same as self.eval, the value is given by the numerica derivative.
        ------------------------------
        """
        
        if variable == 'x':
            variable = 0
        if variable == 'y':
            variable = 1
        if variable == 'z':
            variable = 2
        
        def incr_variable(inc,*args):
            args_new = ()
            for i in range(len(args)):
                if i == variable:
                    args_new += (args[i]+inc,)
                else:
                    args_new += (args[i],)
            return args_new
        
        if mode == 'increment':
            return (self.eval(*incr_variable(h,*args))-self.eval(*args))/h
        elif mode == 'decrement':
            return (self.eval(*args)-self.eval(*incr_variable(-h,*args)))/h
        elif mode == 'symmetric':
            return (self.eval(*incr_variable(h,*args))-self.eval(*incr_variable(-h,*args)))/(2*h)
        elif mode == 'five-point':
            point1 = -self.eval(*incr_variable(2*h,*args))
            point2 = 8*self.eval(*incr_variable(h,*args))
            point3 = -8*self.eval(*incr_variable(-h,*args))
            point4 = self.eval(*incr_variable(-2*h,*args))
            summed = point1 + point2 + point3 + point4
            return summed/(12*h)
        else: 
            print('The selected mode is invalid.')
            return None
        
    def func_pder(self,
                  **kwargs):
        """
        Method for evaluating the n-th derivative
        ---------------
        Parameters:
            args: arguments, float
                where to evaluate the function
            kwargs: key arguments
                the key arguments of function.pder()
        --------------
        Returns:
            function.function: a function class corresponding to the
                first derivative
        -------------

        WARNING: after the third time that is applied, it seems to deliver
        very wrong results. Currently not clear why.
        """

        def func (*args):
            return self.pder(*args,**kwargs)

        return function(func)

    def grad(self,
             *args,
             h = 1e-6,
             mode = 'five-point'):
        """
        Returns the gradient of the function with respect to all its variables.
        -----------------------
        Parameters:
            args: tuple of floats
                parameters passed to self.function() for the numeric derivative
            h: float, optional
                incremental step used for the incremental ratio. Defaults to 1e-6.
            mode: string, optional
                numeric derivative mode, see function.pder() for more documentation.
                Defaults to 'five-point'
        ------------------------
        Returns:
            numpy.ndarray:
                1D array made of the numeric derivatives of self.function with respect 
                to all its variables. If self.function is vectorial (i.e. it returns an 
                array), then every element of the returned 1D array is a vector.
        ------------------------
        """
        
        gradient_list = []
        dimensions = len(args)
        for i in range(0,dimensions):
            gradient_list.append(self.pder(*args,variable = i,h = h, mode = mode))
        return np.array(gradient_list)

    def gradient_descent(self,           
                         initial = None, 
                         maxiter = 1e4,
                         precision = 1e-6,
                         h = 1e-6,
                         step = 1e-2,
                         show_steps = False,
                         mode = 'five-point'):
                         
        """
        Method used to find the global minimum of the function.
        For maximum, just use the opposite self.function
        --------------------
        Parameters:
            initial: list or np.array, 
                initial list of parameters for self.function to start from.
                If None, it is assumed to be null.
            maxiter: int
                maximum number of iterations before returning.
                Used to avoid freezing of the method
            precision: float
                when the algorithm changes the current by less than 'precision',
                the method returns the current position and convergence is achieved.
            h: float
                step used for the incremental ratio
            step: float
                weight applied to the gradient descent
            show_steps: bool, optional
                if True, the method also returns the trajectory followed by the algorithm
                to reach the minimum. Its use is limited to monitoring and debugging.
                Defaults to False
            mode: string, optional
                Method used for the numeric derivative in the gradient. See function.pder() for
                more documentation. Defaults to 'five-point'.
        ------------------------
        Returns:
            if show_steps is True:
                numpy.ndarray:
                    the result of the gradient descent.
            if show_steps is False:
                tuple:
                    the result of the gradient descent and
                    a list of all the intermediate steps.
        --------------------------
        """
                        
        if initial == None:
            initial = np.zeros(self.dimensions)
        x0 = initial
        converged = False
        iteration = 0
        if show_steps:
            steps = []
            steps.append(x0)

        while ((iteration < maxiter) and not converged):
            iteration += 1
            previous = x0
            x0 -= step*self.grad(*x0, h = h, mode = mode)
            converged = (np.sum(np.abs(self.grad(*x0,h=h,mode = mode))) < precision)
            if show_steps:
                steps.append(previous-step*self.grad(*previous, h = h, mode = mode))

        if iteration == maxiter:
            print('No convergence reached before the maximum number of iterations!')
            print('Residual gradient:',self.grad(*x0,h=h,mode = mode))

        if show_steps:
            return (x0, steps)
        else:
            return x0
    
    def least_square_fit(self,          
                         xdata,
                         ydata,
                         **kwargs):
        """
        This function makes use of a closure in order to create a new
        function that gives the sum of squared residues given the
        data sets xdata and ydata. It is always assumed that self.function
        has the first argument as independent variable and the rest as
        parameters.
        -------------------
        Parameters: 
            xdata: numpy.ndarray
                arguments at which self.function has been measured
            ydata: numpy.ndarray
                measured values of self.function
            kwargs: keyword arguments used for self.gradient_descent()
        --------------------
        Returns:
            the returned value of the function.gradient_descent() related
            to the loss function.
        --------------------
        WARNING: It seems not to be working for more than 1 dimension for the moment.
        """

        def loss_function(self, xdata, ydata):
            def internal_loss(*args):
                partial_sum = 0
                for i in range(0,len(xdata)):
                    args_complete = (xdata[i],) + args
                    partial_sum += (ydata[i]-self.eval(*args_complete))**2
                return partial_sum
            return internal_loss

        loss = loss_function(self,xdata,ydata)
        classed_loss = function(loss)
        return classed_loss.gradient_descent(**kwargs)
    
    def find_zero(self,
                  *args,
                  z0 = 0,  
                  tolerance = 1e-12, 
                  method = 'newton',
                  **kwargs):
        """
        Method that uses numerical iterations to find the zeros of the self function.
        The method finds the zero only for one dimensional functions or for functions whose
        other variables have been frozen.
        -----------------
        Parameters:
            args: arguments corresponding to possible additional variables.
            z0: float, optional
                initial condition for the variable of interest.
                Defaults to 0.
            tolerance: float, optional
                indication on when to stop to iterative scheme.
                If the variable changes by an amount smaller than the tolerance,
                the computation is stopped. Defaults to 1e-12.
            method: string, optional
                Method for solving the system. So far, only the 'newton' method has 
                been implemented (Newton-Raphson algorithm).
            kwargs: keyword arguments used for the numeric derivative.
        ----------------
        Returns:
            float: zero of the function.
        ----------------
        """
        
        if method == 'newton':
            
            delta = np.inf
            evaluator = self.eval
            derivator = self.pder
            while delta > tolerance:
                prev = z0
                z0 = z0 - evaluator(z0,*args)/derivator(z0,*args, **kwargs)
                delta = np.abs(prev - z0)
        
        else:
            raise Exception('Method not implemented.')
        
        return z0
    
    @staticmethod
    def x():
        """
        Returns:
            instance of the class function where self.function is the function f(x) = x.
        """
        
        def func(t):
            return t
        return function(func)
        
    # def integral(self,domain):
    
    # def hessian(self):


class diffusion():

    def __init__(self,
                 mu,
                 sigma,
                 alpha = 0.0,
                 state_variables = 1,
                 time_dependent = False):
        """
        ----------------------
        Parameters:
            mu: type function or class function (see above)
                represents the drift of the diffusion (in Ito's sense)
            sigma: type function of class function (see above)
                represents the volatility of the diffusion
            alpha: float, list or numpy.ndarray, optional
                value (or matrix) corresponding to the stochastic parameter.
                When in matrix form, the corresponding multidimensional 
                diffusion has a hybrid stochastic integration convention.
                Defaults to 0.0
            state_variables: int
                the number of state variables. This parameter is used when
                determining the number of dimensions. Specifically, if alpha
                is a vector, then it is a column if state_variables > 1 and it 
                is a row if state_variables == 1.
                Defaults to 1.
            time_dependent: bool, optional
                When True, the first argument of self.mu and self.sigma is 
                the time. Defaults to False
                WARNING: so far no implemented
        ----------------------
        """
                
        self.drift = function(mu) if type(mu) == type(foo) else mu
        self.volatility = function(sigma) if type(sigma) == type(foo) else sigma
        self.state_variables = state_variables
        self.dimension = None   # self.set_alpha() initializes this argument
        self.alpha = None       # self.set_alpha() initializes this argument
        self.set_alpha(alpha = alpha)
        self.time_dependent = time_dependent

    def get_alpha(self):
        """
        ------------
        Returns:
            The method returns a scalar if the problem is one
            dimensional, it returns a 1D numpy array if the problem is vectorial 
            (one state variable) and a 2D numpy array if the problem is multidimensional.
        ------------
        """

        if len(self.alpha.shape) == 0:
            return self.alpha + 0.0
        elif len(self.alpha.shape) == 1:
            return self.alpha
        else:
            if self.alpha.shape == (1,1):
                return self.alpha[0,0]
            elif self.alpha.shape[0] == 1 ^ self.alpha.shape[1] == 1:
                return self.alpha[0] if self.alpha.shape[1] == 1 else self.alpha[1]
            else:
                return self.alpha

    def set_dimension(self):
        """
        Method that uses the dimensionality of alpha to set the dimensionality
        of the diffusion itself. Does not return anything, only updates the value
        of self.dimension.
        """

        alpha = self.get_alpha()
        if type(alpha) is float:
            dim = (1, 1)
        else:
            if len(self.alpha.shape) == 1:
                if self.state_variables == 1:
                    dim = (1, len(self.alpha))
                else:
                    dim = (len(self.alpha),1)
            elif len(self.alpha.shape) == 0:
                dim = (1,1)
            else:
                dim = self.alpha.shape
        self.dimension = dim

    def set_alpha(self,
                  alpha = 0.0):
        """
        --------------
        Parameters:
            alpha: float, list or numpy.ndarray, optional
                The stochastic parameter to be applied to the diffusion.
                Defaults to 0.0
        -----------------
        """
        
        self.alpha = np.array(alpha)
        self.set_dimension()

    def realization(self,                   
                    x0 = None,
                    step = 1e-7,
                    length = 1e-2,
                    return_time = False,
                    alpha_loc = None,
                    h = 1e-7,
                    mode = 'five-point'):
        """
        This method returns a realization of the diffusion 
        dx = self.drift*dt + self.volatility*dW,
        where W is a Wiener process.
        ----------------------
        Parameters:
            x0: float or numpy.ndarray, optional
                initial state of the diffusion.
                Defaults to 0.0
            step: float, optional
                length of the time increments when solving numerically
                the stochastic diffusion. Defaults to 1e-7.
            length: float, optional
                total length of the realization. length//step corresponds
                to the number of elements in the returned numpy.ndarray.
                Defaults to 1e-2.
            return_time: bool, optional 
                If True, the vector corresponding to time is also returned.
                Defaults to False
            alpha_loc: float or list or numpy.ndarray, optional
                needed in case the desired stochastic parameter for this specific
                realization does not coincide with self.alpha.
                If None, self.get_alpha() is used.
            h: float, optional
                incremental step used for the numeric derivative, see function.pder()
                Defaults to 1e-7.
            mode: string, optional
                Method used for the numeric derivative, see function.pder() for more 
                documentation. Defaults to 'five-point'
        ------------------------
        Returns:
            if return_time:
                tuple: 
                    list corresponding to the time axis
                    list corresponding to the states occupied by the diffusion
            else:
                list corresonding to the states occupied by the diffusion
        ------------------------
        """
        
        # check that the seed is always random!
        
        if alpha_loc == None:
            alpha_loc = self.get_alpha()

        dimension = self.dimension    
        if x0 is None:
            x0 = np.zeros(dimension[0])
        x = []
        if len(x0) > 1:
            x.append(x0.copy())
        else:
            x.append(x0)
        t = 0.0
        steps = int(length/step)
        if return_time:
            time = []
            time.append(t)
            
        print('---------------------')
        print('Constructing realization...')
        
        for current_step in tqdm(range(steps)):       
            t += step
            if return_time:
                time.append(t)
            dW = np.random.normal(size = dimension[1], scale = np.sqrt(step))

            if not self.time_dependent:
                if dimension != (1, 1):
                    sigma = self.volatility.eval(*x0)

                    def alphasigma(*args):
                        sig = self.volatility.eval(*args)
                        return alpha_loc * sig
                    alpha_sigma = function(alphasigma)
                    csi = np.zeros(dimension[0])
                
                    # correction term csi
                    for i in range(dimension[0]):
                        for k in range(dimension[0]):
                            for j in range(dimension[1]):
                                csi[i] += sigma[k,j] * alpha_sigma.pder(*x0,
                                                                        variable = k,
                                                                        h = h,
                                                                        mode = mode)[i,j]
                    x0 += (self.drift.eval(*x0) + csi)*step + np.dot(self.volatility.eval(*x0),dW)
                else:
                    mu_drift = np.array([self.drift.eval(*x0)])
                    sigma_drift = alpha_loc*self.volatility.eval(*x0)*self.volatility.pder(*x0,
                                                                                           h = h,
                                                                                           mode = mode)
                    sigma_drift = np.array([sigma_drift])
                    sigma_diff = self.volatility.eval(*x0)*dW
                    x0 += (mu_drift + sigma_drift) * step + sigma_diff
            else:
                xt0 = (x0, t)

                # correction term csi
                for i in range(dimension[0]):
                    for k in range(dimension[0]):
                        for j in range(dimension[1]):
                            csi[i] = sigma[k,j] * alpha_sigma.pder(*xt0,
                                                                   variable = k,
                                                                   h = h,
                                                                   mode = mode)[i,j]
                x0 += (self.drift.eval(*xt0) + csi)*step + np.dot(self.volatility.eval(*xt0),dW)
            if len(x0) > 1:
                x.append(x0.copy())
            else:
                x.append(x0[0])
                
        print('Realization completed')
        print('---------------------')
            
        return (np.array(x),np.array(time)) if return_time else np.array(x)
    
    def ensemble_average(self,
                         quantity = function.x(),
                         save_plot = False,
                         num_realizations = 100,
                         shown_plots = 20,
                         lw = 2,
                         fs = 14,
                         figs = (6,6),
                         time_unit = '[s]',
                         path = '../saved_plots/ensemble_sum.png',
                         **kwargs):
        """
        Calculate E[quantity] given the diffusion dx = self.drift*dt + self.volatility*dW
        the result is a function of time, since x itself it is.
        ----------------------
        Parameters:
            quantity: class function, optional
                the function whose ensemble average over many realization of the diffusion
                needs to be calculated. Defaults to function.x(), corresponding to E[x].
            save_plot: bool, optional
                If true, a png file with realizations and ensemble average is saved 
                in a location corresponding to path. Defaults to False
            num_realizations: int, optional
                number of realizations that need to be computed before performing the
                ensemble average. Defaults to 100
            shown_plots: int, optional
                used when save_plot is True. In the first subplot, where different 
                realizations are shown as an example, shown_plots determines how many.
                Defaults to 20.
            lw: float, optional
                used when save_plot is True. It determines the linewidth used for the plots.
                Defaults to 2.
            fs: float, optional
                used when save_plot is True. It determines the fontsize used for the labels.
                Defaults to 14.
            figs: tuple of two floats, optional
                used when save_plot is True. It determines the size of the figure.
                Defaults to (6,6).
            time_unit: string, optional
                used when save_plot is True. It determines the unit used for the xlabel.
                Defaults to '[s]'.
            path: string, optional
                Location where the desired file needs to be saved.
                Defaults to '../saved_plots/ensemble_sum.png'.
            kwargs: keyword arguments that are passed to function.realization.
        ------------------------
        Returns:
            tuple of lists:
                first element given by the ensemble average
                second element given by the ensemble variance
        ------------------------
        """
        
        if 'return_time' in kwargs:
            kwargs['return_time'] = False
        if save_plot:
            step = kwargs['step'] if ('step' in kwargs) else 1e-7
            length = kwargs['length'] if ('length' in kwargs) else 1e-2

        realizations = []
        for i in tqdm(range(num_realizations)):
            temp = self.realization(**kwargs)
            for i in range(len(temp)):
                temp[i] = quantity.eval(*temp[i])
            realizations.append(temp)
        timetrace_length = len(realizations[0])
        (ensemble,variance) = diffusion.ensemble_sum(realizations)
        if save_plot:
            time = np.linspace(0,length,timetrace_length)
            diffusion.save_plot(time,realizations[:shown_plots],
                                ensemble,variance,lw = lw,fs = fs,path = path,figs = figs)
        return (ensemble, variance)
    
    @staticmethod
    def save_plot(time,
                  realizations,
                  ensemble,
                  variance,
                  lw = 2,
                  fs = 14,
                  path = '../saved_plots/ensemble_plot',
                  figs = (6,6),
                  time_unit = '[s]'):
        """
        Method that creates a .png file that represents the realizations and the
        corresponding ensemble average.
        -----------------------------
        Parameters:
            time: indexed array of floats
                the time axis corresponding to the realizations.
            realizations: list of indexed arrays of floats 
                the realizations that need to appear in the first subplot
                as examples of the ones that contributed to the ensemble average.
            ensemble: indexed array of floats
                the ensemble average of the realizations.
            variance: indexed array of floats
                the ensemble variance of the realizations
            lw: float, optional 
                linewidth used for the plots. Defaults to 2.
            fs: float, optional
                fontsize used for the labels. Defaults to 14.
            path: string, optional
                location of the saved file.
                Defaults to '../saved_plots/ensemble_sum.png'.
            figs: tuple of two floats, optional
                Size of the figure, used in plt.figure().
                Defaults to (6,6)
            time_unit: string, optional
                Unit of measurement of the time used for the xlabel of the figure.
        ------------------------------
        """
        
        fig,[ax1,ax2] = plt.subplots(nrows = 2,ncols = 1,sharex = True,figsize = figs)
        for i in range(len(realizations)):
            ax1.plot(time,realizations[i],linewidth = 0.5)
        ax2.plot(time,ensemble,linewidth = lw,label = 'Average')
        ax2.fill_between(time,ensemble+np.sqrt(variance),
                    ensemble-np.sqrt(variance),alpha = 0.3,facecolor='b',edgecolor='b')
        ax2.set_xlabel('Time '+time_unit,fontsize = fs)
        ax1.set_ylabel('Realizations',fontsize = fs)
        ax2.set_ylabel('Ensemble',fontsize = fs)
        fig.savefig(path)

    @staticmethod
    def ensemble_sum(realizations):
        """
        Caluculates the ensemble over the given realizations.
        ------------------
        Parameters:
            realizations: list of indexed arrays
                the realizations whose ensemble average needs to be
                computed.
        -----------------
        Returns:
            tuple of two lists:
                first element is the ensemble average
                second element is the ensemble variance
        -----------------
        """

        timetrace_length = len(realizations[0])
        ensemble = np.zeros(timetrace_length)
        variance = np.zeros(timetrace_length)
        num_realizations = len(realizations)
        for i in range(num_realizations):
            ensemble += realizations[i]
        ensemble /= num_realizations
        for i in range(num_realizations):
            variance += (realizations[i]-ensemble)**2
        variance /= (num_realizations-1)
        return (ensemble,variance)
                
    def estimate_alpha(self,
                       realization,
                       grid = np.linspace(-100e-9,100e-9,250),
                       step = 1e-7,
                       verbose = False):
        """
        given a realization corresponding to the process "self", it estimates the 
        stochastic parameter alpha used.
        It is assumed that the stochastic parameter is state-independent.
        -------------------
        Parameters:
            realization: indexed array 
                the realization used to estimate the stochastic parameter.
            grid: numpy.ndarray, optional
                1D array of the points where the stochastic parameter needs to be calculated.
                Defaults to np.arange(-100e-9,100e-9,250).
            step: float, optional
                Time steps between different samples of the realization.
                Defaults to 1e-7
            verbose: bool, optional
                if False, only the estimated stochastic parameter is returned.
                If True, the stochastic parameter, the drifts and the diffusions used to
                calculate it are returned as well. Defaults to False.
        --------------------
        Returns:
            If verbose:
                tuple: six elements
                    - first element is the estimated stochastic parameter from
                        an average over all the positions where it has been estimated.
                        Given as (mean,standard deviation)
                    - second element is a list of the estimated stochastic parameter 
                        at various positions.
                        Given as (mean,standard deviation)
                    - third element is a list of the positions where the stochatic
                        parameter has been estimated.
                    - fourth element is a list of the estimated drifts at various
                        positions along the grid.
                        Given as (mean, standard deviation)
                    - fifth element is a list of all the estimated diffusions
                        at various positions along the grid.
                        Given as (mean, standard deviation)
                    - sixth element is a list of all the positions at which
                        drifts and diffusions have been estimated.
            else:
                float: 
                    estimated value of the stochastic parameter.
        --------------------
        WARNING: Method devised only for the one dimensional diffusion case
        """
                
        samples = len(realization)
        estimations = len(grid)

        drifts = {i : [] for i in range(estimations)}
        Edrifts = {i : None for i in range(estimations)}
        diffusions = {i : [] for i in range(estimations)}
        Ediffusions =  {i : None for i in range(estimations)}
        alphas =  {i : None for i in range(estimations)}
        
        print('Setting grid...')
        
        for i in tqdm(range(samples-1)):
            position = put_in_grid(grid, realization[i], 'binary')
            if position is not None:
                drifts[position].append((realization[i+1]-realization[i])/step)
                diffusions[position].append(1/2*(realization[i+1]-realization[i])**2/step)
        
        print('Grid preparation complete.')
        print('-------------------------')
        print('Estimating parameters...')
        
        meaningful_elements = []    # contains indexes where calculation makes sense
        alpha = 0
        
        for i in tqdm(range(estimations)):
            if len(drifts[i]) > 1:
                meaningful_elements.append(i)

                mean_drift = np.sum(drifts[i])/len(drifts[i])
                var_drift = np.sum((drifts[i]-mean_drift)**2/(
                                            (len(drifts[i]))*(len(drifts[i])-1)))

                Edrifts[i] = (mean_drift,np.sqrt(var_drift))

                mean_diffusion = np.sum(diffusions[i])/len(diffusions[i])
                var_diffusion = np.sum((diffusions[i]-mean_diffusion)**2/(
                                    (len(diffusions[i]))*(len(diffusions[i])-1)))

                Ediffusions[i] = (mean_diffusion,np.sqrt(var_diffusion))
                
        print('Parameters have been estimated.')
        print('-------------------------')
        print('Calculating the stochastic parameter...')
        
        meaningful_alphas = []  # contains indexes where numerical derivative of sigma makes sense

        for position in tqdm(meaningful_elements):
            if (position + 1) in meaningful_elements:

                meaningful_alphas.append(position)

                # average value of alpha at position grid[position]
                mu = self.drift.eval(grid[position])
                derD = (Ediffusions[position+1][0]-Ediffusions[position][0])/step
                mean_alpha = (Edrifts[position][0] - mu)/derD

                # error propagation from the variances of the drift and diffusion
                delta_derD = Ediffusions[position+1][1] + Ediffusions[position][1]
                delta_mu = Edrifts[position][1]
                frac_derD = delta_derD/derD
                frac_mu = delta_mu/mu
                std_alpha = mean_alpha * np.sqrt(frac_derD**2 + frac_mu**2)

                # alpha estimation with corresponding error bar
                alphas[position] = (mean_alpha, std_alpha)

        print('Stochastic parameter has been estimated.')
        print('-------------------------')

        if len(meaningful_alphas) == 0:
            print('Need to redefine grid, the stochastic parameter could not be estimated')
        else:
            N = len(meaningful_alphas)
            alpha_i = np.array([alphas[i][0] for i in meaningful_alphas])
            delta_alpha_i = np.array([alphas[i][1] for i in meaningful_alphas])
            normalization = np.sum(1/delta_alpha_i**2)
            ai = 1/(delta_alpha_i**2 * normalization)
            alpha_mean = np.sum(ai*alpha_i)
            var_alpha = np.sum(ai**2 * delta_alpha_i**2)
            alpha = (alpha_mean, var_alpha)

        if verbose:            
            used_drifts = [Edrifts[i] for i in meaningful_elements]
            used_diffusions = [Ediffusions[i] for i in meaningful_elements]
            used_positions = [grid[i] for i in meaningful_elements]
            used_alphas = [alphas[i] for i in meaningful_alphas]
            used_alpha_positions = [grid[i] for i in meaningful_alphas]

        return alpha if not verbose else (alpha,
                                          used_alphas,
                                          used_alpha_positions,
                                          used_drifts,
                                          used_diffusions,
                                          used_positions)
                                          
    def plot_D_estimations(self,
                           estimations,
                           figsize = (8,8),
                           fontsize = 24,
                           save_file = False,
                           superpose_theory = True,
                           include_variance = False,
                           name = 'saved_plot'):
        """
        Function for plotting the estimation returned by diffusion.estimate_alpha.
        --------------------
        Parameters:
            estimations: tuple
                The tuple returned by diffusion.estimate_alpha when verbose is True.
            figsize: tuple, optional
                Size of the figure (width, height). Defaults to (8,8).
            fontsize: int, optional
                Size of the fonts used for the labels and for the ticks.
                Defaults to 24.
            save_file: bool, optional
                Defaults to False
            superpose_theory: bool, optional
                If True, a plot of the theoretical drift and volatility of self
                is included. Defaults to True.
            include_variance: bool, optional
                If True, the plot of the drift and of the diffusion will be surrounded
                by the variance of the estimation. Defaults to False.
            name: str, optional
                Used when save_file is True. The extension for both .pdf and
                .png is automatically taken into account.
                Defaults to 'save_plot' in the same folder.
        ---------------------
        """
        (alpha,
         used_alphas,
         used_alpha_positions,
         used_drifts,
         used_diffusions,
         used_positions) = estimations

        pos = np.array([used_positions[i] for i in range(len(used_positions))])
        drifts = np.array([used_drifts[i][0] for i in range(len(used_positions))])
        theory_drifts = np.array([self.drift.eval(pos[i]) for i in range(len(pos))])
        std_drifts = np.array([used_drifts[i][1] for i in range(len(used_positions))])
        drifts_up = drifts + std_drifts
        drifts_down = drifts - std_drifts
        diffs = np.array([used_diffusions[i][0] for i in range(len(used_positions))])

        if self.dimension[1] == 1:
            theory_diffs = 1/2*(np.array([self.volatility.eval(pos[i]) for i in range(len(pos))]))**2
        else:
            theory_diffs = np.zeros(len(pos))
            sigma_array = [self.volatility.eval(pos[i]) for i in range(len(pos))]
            for i in range(len(pos)):
                try:
                    theory_diffs[i] = 1/2*np.dot(sigma_array[i][0],sigma_array[i][0])
                except:
                    theory_diffs[i] = 1/2*np.dot(sigma_array[i],sigma_array[i])
                
        std_diffs = np.array([used_diffusions[i][1] for i in range(len(used_positions))])
        diffs_up = diffs + std_diffs
        diffs_down = diffs - std_diffs

        fig = plt.figure(figsize = figsize)
        ax2 = fig.add_subplot(211)
        ax3 = fig.add_subplot(212)

        ax2.plot(pos/1e-9,drifts/1e-3, 'C0.', linewidth = 2, label = 'estimated')
        if superpose_theory:
            ax2.plot(pos/1e-9,theory_drifts/1e-3, 'C1--', linewidth = 2,label = 'analytical')
        if include_variance:
            ax2.fill_between(pos/1e-9,drifts_up/1e-3,drifts_down/1e-3, alpha = 0.5)

        fss = fontsize
        xlabel = 'Position [nm]'
        ylabel1 = r'Drift [$\mu$m/ms]'
        ylabel2 = r'Diffusion [$\mu$m$^2$/ms]'

        ax3.plot(pos/1e-9, diffs/1e-9, 'C0.', linewidth = 2, label = 'estimated')
        if superpose_theory:
            ax3.plot(pos/1e-9, theory_diffs/1e-9, 'C1--', linewidth = 2, label = 'analytical')
        if include_variance:
            ax3.fill_between(pos/1e-9, diffs_up/1e-9, diffs_down/1e-9,alpha = 0.5)

        set_ax(ax2,fs = fss, ylabel = ylabel1, legend = True)
        set_ax(ax3,fs = fss, ylabel = ylabel2, xlabel = xlabel, legend = True)

        plt.tight_layout()
        if save_file:
            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png')
        plt.show()

    def plot_array_D_estimations(self,
                                 estimations_tuple,
                                 figsize=(8, 8),
                                 fontsize=24,
                                 save_file=False,
                                 labels = None,
                                 name='saved_plot'):
        """
        Function for plotting the estimation returned by diffusion.estimate_alpha.
        --------------------
        Parameters:
            estimations_tuple: tuple of tuples
                The tuple returned by diffusion.estimate_alpha. Every element of
                estimations_tuple corresponds to such a tuple.
            fontsize: int, optional
                Size of the fonts used for the labels and for the ticks.
                Defaults to 24.
            save_file: bool, optional
                Defaults to False
            save_file: bool, optional
                Defaults to False
            labels: tuple of str, optional
                A legend is included for the realizations. The elements of labels
                correspond to the label of each realization. If none, the index of the
                tuple will be used.
            name: str, optional
                Used when save_file is True. The extension for both .pdf and
                .png is automatically taken into account.
                Defaults to 'save_plot' in the same folder.
        ---------------------
        """

        realizations = len(estimations_tuple)

        if labels is None:
            labels = [str(i) for i in range(realizations)]

        pos = []
        drifts = []
        diffs = []

        for indx in range(realizations):
            current_estimations = estimations_tuple[indx]
            used_drifts = current_estimations[3]
            used_diffusions = current_estimations[4]
            used_positions = current_estimations[5]

            pos.append(np.array([used_positions[i] for i in range(len(used_positions))]))
            drifts.append(np.array([used_drifts[i][0] for i in range(len(used_positions))]))
            diffs.append(np.array([used_diffusions[i][0] for i in range(len(used_positions))]))

        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        for realization in range(realizations):
            ax0.plot(pos[realization] / 1e-9,
                     drifts[realization] / 1e-3,
                     '.',
                     linewidth=2,
                     label=labels[realization])

            ax1.plot(pos[realization] / 1e-9,
                     diffs[realization] / 1e-9,
                     '.',
                     linewidth=2,
                     label = labels[realization])

        fss = fontsize
        xlabel = 'Position [nm]'
        ylabel1 = r'Drift [$\mu$m/ms]'
        ylabel2 = r'Diffusion [$\mu$m$^2$/ms]'

        set_ax(ax0, fs=fss, ylabel=ylabel1, legend=True)
        set_ax(ax1, fs=fss, ylabel=ylabel2, xlabel=xlabel, legend=True)

        plt.tight_layout()
        if save_file:
            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png')
        plt.show()

    def plot_nb_D_estimations(self,
                              estimations,
                              figsize=(8, 8),
                              fontsize=24,
                              save_file=False,
                              superpose_theory=True,
                              name='saved_plot'):
        """
        Function for plotting the estimation returned by diffusion.estimate_alpha.
        --------------------
        Parameters:
            estimations: tuple
                The tuple returned by get_D_nb when verbose is True.
            figsize: tuple, optional
                Size of the figure (width, height). Defaults to (8,8).
            fontsize: int, optional
                Size of the fonts used for the labels and for the ticks.
                Defaults to 24.
            save_file: bool, optional
                Defaults to False
            superpose_theory: bool, optional
                If True, a plot of the theoretical drift and volatility of self
                is included. Defaults to True.
            name: str, optional
                Used when save_file is True. The extension for both .pdf and
                .png is automatically taken into account.
                Defaults to 'save_plot' in the same folder.
        ---------------------
        """

        pos = estimations[0]
        drifts = estimations[1]
        diffs = estimations[2]
        theory_drifts = np.array([self.drift.eval(pos[i]) for i in range(len(pos))])

        if self.dimension[1] == 1:
            theory_diffs = 1 / 2 * (np.array([self.volatility.eval(pos[i]) for i in range(len(pos))])) ** 2
        else:
            theory_diffs = np.zeros(len(pos))
            sigma_array = [self.volatility.eval(pos[i]) for i in range(len(pos))]
            for i in range(len(pos)):
                try:
                    theory_diffs[i] = 1 / 2 * np.dot(sigma_array[i][0], sigma_array[i][0])
                except:
                    theory_diffs[i] = 1 / 2 * np.dot(sigma_array[i], sigma_array[i])

        fig = plt.figure(figsize=figsize)
        ax2 = fig.add_subplot(211)
        ax3 = fig.add_subplot(212)

        ax2.plot(pos / 1e-9, drifts / 1e-3, 'C0.', linewidth=2, label='estimated')
        if superpose_theory:
            ax2.plot(pos / 1e-9, theory_drifts / 1e-3, 'C1--', linewidth=2, label='analytical')

        fss = fontsize
        xlabel = 'Position [nm]'
        ylabel1 = r'Drift [$\mu$m/ms]'
        ylabel2 = r'Diffusion [$\mu$m$^2$/ms]'

        ax3.plot(pos / 1e-9, diffs / 1e-9, 'C0.', linewidth=2, label='estimated')
        if superpose_theory:
            ax3.plot(pos / 1e-9, theory_diffs / 1e-9, 'C1--', linewidth=2, label='analytical')

        set_ax(ax2, fs=fss, ylabel=ylabel1, legend=True)
        set_ax(ax3, fs=fss, ylabel=ylabel2, xlabel=xlabel, legend=True)

        plt.tight_layout()
        if save_file:
            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png')
        plt.show()
        
    @staticmethod
    def plot_nb_array_D_estimations(estimations_tuple,
                                 figsize=(8, 8),
                                 fontsize=24,
                                 save_file=False,
                                 labels = None,
                                 name='saved_plot'):
        """
        Function for plotting the estimation returned by diffusion.estimate_alpha.
        --------------------
        Parameters:
            estimations_tuple: tuple of tuples
                The tuple returned by get_D_nb. Every element of
                estimations_tuple corresponds to such a tuple.
            fontsize: int, optional
                Size of the fonts used for the labels and for the ticks.
                Defaults to 24.
            save_file: bool, optional
                Defaults to False
            save_file: bool, optional
                Defaults to False
            labels: tuple of str, optional
                A legend is included for the realizations. The elements of labels
                correspond to the label of each realization. If none, the index of the
                tuple will be used.
            name: str, optional
                Used when save_file is True. The extension for both .pdf and
                .png is automatically taken into account.
                Defaults to 'save_plot' in the same folder.
        ---------------------
        """

        realizations = len(estimations_tuple)

        if labels is None:
            labels = [str(i) for i in range(realizations)]

        pos = []
        drifts = []
        diffs = []

        for indx in range(realizations):
            current_estimations = estimations_tuple[indx]
            pos.append(current_estimations[0])
            drifts.append(current_estimations[1])
            diffs.append(current_estimations[2])

        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        for realization in range(realizations):
            ax0.plot(pos[realization] / 1e-9,
                     drifts[realization] / 1e-3,
                     '.',
                     linewidth=2,
                     label=labels[realization])

            ax1.plot(pos[realization] / 1e-9,
                     diffs[realization] / 1e-9,
                     '.',
                     linewidth=2,
                     label = labels[realization])

        fss = fontsize
        xlabel = 'Position [nm]'
        ylabel1 = r'Drift [$\mu$m/ms]'
        ylabel2 = r'Diffusion [$\mu$m$^2$/ms]'

        set_ax(ax0, fs=fss, ylabel=ylabel1, legend=True)
        set_ax(ax1, fs=fss, ylabel=ylabel2, xlabel=xlabel, legend=True)

        plt.tight_layout()
        if save_file:
            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png')
        plt.show()

#TODO:
# add higher order derivatives
# class probability_density(function): 
# overload operators inside the diffusion class, fulfilling Ito's formula

