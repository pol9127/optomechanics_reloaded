'''
@ author: Andrei Militaru
@ date: 18th of November 2019
'''

import numpy as np
from math import factorial
from tqdm import tqdm


class GaussianMode():
    
    def __init__(self, 
                 waist=5e-3, 
                 region=(-50e-3, 50e-3), 
                 origin=0, 
                 resolution=100,
                 wavelength=1064e-9,
                 amplitude=1):
        '''
        Class that creates and propagates a fundamental gaussian mode.
        Such mode is the base for all higher transverse modes.
        All units are in SI system.
        ----------------
        Parameters:
            waist: float, optional
                beam 1e-2 radius. 
                Defaults to 5e-3
            region: tuple of floats
                Region of space over which the beam is defined.
                If the length is two, the same profile is assumed for both
                transverse directions.
                Defaults to (-50e-3, 50e-3).
            origin: float, optional
                Position of the beam waist with respect to the optical axis.
                Defaults to 0.
            resolution: int or tuple of int, optional
                Number of point where the profile is evaluated.
                If int, the same resolution is assumed for both transverse direction.
                Defaults to 100.
            wavelength: float, optional
                Wavelength of the beam considered
            amplitude: numpy complex type, optional
                Complex amplitude of the beam.
                Defaults to 1.
        -----------------
        '''
                
        if type(resolution) is not tuple:
            resolution = (resolution, resolution)
        if len(region) == 2:
            region = (region[0], region[1], region[0], region[1])
        self.waist = waist
        self.x = np.linspace(region[0], region[1], resolution[0])
        self.y = np.linspace(region[2], region[3], resolution[1])
        self.wavelength = wavelength
        self.origin = origin
        self.resolution = resolution
        self.region = region
        self.amplitude = amplitude

    def Zrayleigh(self):
        '''
        Returns the Rayleigh range of the beam.
        -----------------
        Returns:
            float: Rayleigh range of beam [m].
        -----------------
        '''
        return np.pi*self.waist**2/self.wavelength

    def width(self, z):
        '''
        Returns the 1e-2 width as a function of position along optical axis.
        --------------
        Parameters:
            z: float
                Position along the optical axis [m].
        --------------
        '''
        Dz = z - self.origin
        return self.waist*np.sqrt(1+Dz**2/self.Zrayleigh()**2)

    def divergence(self):
        '''
        Returns the angular coefficient of the divergence of the beam in far field.
        ----------------
        Returns:
            float: divergence angular coefficient.
        ----------------
        '''
        return self.waist/self.Zrayleigh()

    def radius(self, z):
        '''
        Returns the radius of curvature of the phase of the beam.
        -----------------
        Parameters:
            z: float
                Position along the optical axis [m].
        -----------------
        Returns:
            float: the radius of curvature [m].
        ----------------
        '''
        Dz = z - self.origin
        try:
            return Dz*(1 + (self.Zrayleigh()/Dz)**2)
        except:
            return np.inf

    def gouy(self, z, N=0):
        '''
        Returns the Gouy phase as a function of the position along the optical axis.
        ----------------
        Parameters:
            z: float
                Position along the optical axis [m].
            N: int, optional
                Order of the higher mode. For hermitian profiles Hmn, N = m + n.
                Defaults to 0.
        ----------------
        Returns:
            float: Gouy phase of fundamental beam [rad].
        ----------------
        '''
        Dz = z - self.origin
        return (1+N) * np.arctan(Dz/self.Zrayleigh())

    def k(self):
        '''
        --------------
        Returns:
            float: the wavenumber of the beam [m-1].
        --------------
        '''
        return 2*np.pi/self.wavelength

    def gaussian_profile(self, z, N=0):
        '''
        Returns a two dimensional profile of the beam along the optical axis.
        ---------------
        Parameters:
            z: float
                Position along the optical axis [m].
            N: int, optional
                Order of the higher mode. For hermitian profiles Hmn, N = m + n.
                Defaults to 0.
        ---------------
        Returns:
            numpy.ndarray: two-dimensional array of complex numbers indicating
                           the complex field value on the XY plane.
        --------------
        '''
        X,Y = self.grid()
        k = self.k()
        psi = self.gouy(z)
        R = self.radius(z)
        w = self.width(z)
        w0 = self.waist
        sqrR = X**2 + Y**2
        E0 = self.amplitude
        return E0 * w0/w * np.exp(-sqrR/w**2) * np.exp(1j*psi) * np.exp(-1j*k*sqrR/(2*R))

    def grid(self):
        '''
        Returns the mesh grid used to compute the profile of the gaussian beam.
        --------------
        Returns:
            numpy.ndarray: meshgrid of the field.
        --------------
        '''
        return np.meshgrid(self.x, self.y)


class HermiteMode(GaussianMode):

    def __init__(self, order=0, *args, **kwargs):
        '''
        -------------
        Parameters:
            order: int or tuple of int, optional
                Order of the Hermite beam. If int, the same order is assumed
                for both transversal directions.
                Defaults to 0.
            args: arguments that need to be inserted for the GaussianBeam.
            kwargs: key arguments that need to be inserted for the GaussianBeam.
        -------------
        '''
        super().__init__(*args, **kwargs)
        if type(order) is not tuple:
            order = (order, order)
        self.order = order

    @staticmethod
    def polynomial(x, n):
        '''
        Returns the value of the n-th order Hermite polynomial evaluated at position x.
        -------------
        Parameters:
            x: numpy.ndarray
                the position or the grid over which the polynomial needs to be evaluated.
            n: int
                Order of the Hermite polynomial.
        -------------
        Returns:
            numpy.ndarray of same size as x: value of the Hn evaluated over x.
        -------------
        '''
        if n % 2 == 0:
            out = np.zeros_like(x)
            extreme = int(n/2)
            for index in range(extreme + 1):
                out += factorial(n)*(-1)**(extreme - index)/(
                    factorial(2*index)*factorial(extreme - index))*(2*x)**(2*index)
            return out
        else:
            out = np.zeros_like(x)
            extreme = int((n-1)/2)
            for index in range(extreme + 1):
                out += factorial(n)*(-1)**(extreme-index)/(
                    factorial(2*index + 1)*factorial(extreme - index))*(2*x)**(2*index + 1)
            return out

    def project(self, other_profile, z=0):
        '''
        Method that performs the scalar product between a given profile and the mode.
        -------------
        Parameters:
            other_profile: numpy.ndarray of dtype = 'complex'
                Profile that needs to be projected over the mode.
                It is important the it has the same size of the profile of the mode.
            z: float, optional
                Position along the optical axis where other_profile has been evaluated.
                Defaults to z=0.
        -------------
        Returns:
            Instance of HermiteMode with amplitude given by the projection.
        -------------
        '''
        self.amplitude = 1
        profile = self.profile(z=z)
        length_squared = np.sum(np.abs(profile)**2)
        projection = np.sum(np.conj(profile)*other_profile)
        self.amplitude = projection/length_squared
        return self

    def profile(self, z=0, include_grid=False):
        ''' 
        Builds the profile of the mode along the optical axis.
        ---------------
        Parameters:
            z: float, optional
                Position along the optical axis [m].
                Defaults to 0.
            include_grid: bool, optional
                If True, the meshgrid of the positions is returned.
                Defaults to False.
        ---------------
        Returns:
            if include_grid:
                X, Y, profile
            else:
                profile
        ---------------
        '''
        w = self.width(z)
        X,Y = np.meshgrid(self.x, self.y)
        Hn = self.polynomial(np.sqrt(2)*X/w, self.order[0])
        Hm = self.polynomial(np.sqrt(2)*Y/w, self.order[1])
        N = self.order[0] + self.order[1]
        if not include_grid:
            return self.gaussian_profile(z, N=N) * Hn * Hm
        else:
            return X, Y, self.gaussian_profile(z, N=N) * Hn * Hm


class LaguerreMode(GaussianMode):

    def __init__(self, order=0, *args, **kwargs):
        '''
        -------------
        Parameters:
            order: int or tuple of int, optional
                Order of the Laguerre beam. If int, the same order is assumed
                for both transversal directions. 
                order[0] is the radial mode. 
                order[1] is the azimuthal mode.
                Defaults to 0.
            args: arguments that need to be inserted for the GaussianBeam.
            kwargs: key arguments that need to be inserted for the GaussianBeam.
        -------------
        '''
        super().__init__(*args, **kwargs)
        if type(order) is not tuple:
            order = (order, order)
        self.order = order
    
    @staticmethod
    def binomial_coefficient(n,k):
        '''
        -------------
        Returns:
            int: n!/(k!(n-k)!)
        -------------
        '''
        return factorial(n)/(factorial(k)*factorial(n-k))

    @staticmethod
    def polynomial(x, p, l):
        '''
        Returns the value of the Laguerre generalized polynomial
        of order p, l evaluated at position x.
        -------------
        Parameters:
            x: numpy.ndarray
                the position or the grid over which the polynomial needs to be evaluated.
            p: int
                Radial order of the generalized Laguerre polynomial.
            l: int
                Azimuthal order of the generalized Laguerre polynomial.
        -------------
        Returns:
            numpy.ndarray of same size as x: value of the L^l_p evaluated over x.
        -------------
        '''
        output = np.zeros_like(x)
        for i in range(p+1):
            sign = (-1)**i
            coef = LaguerreMode.binomial_coefficient(p+l, p-i)/factorial(i)
            output += sign*coef*x**i
        return output

    def project(self, other_profile, z=0):
        '''
        Method that performs the scalar product between a given profile and the mode.
        -------------
        Parameters:
            other_profile: numpy.ndarray of dtype = 'complex'
                Profile that needs to be projected over the mode.
                It is important the it has the same size of the profile of the mode.
            z: float, optional
                Position along the optical axis where other_profile has been evaluated.
                Defaults to z=0.
        -------------
        Returns:
            Instance of HermiteMode with amplitude given by the projection.
        -------------
        '''
        self.amplitude = 1
        profile = self.profile(z=z)
        length_squared = np.sum(np.abs(profile)**2)
        projection = np.sum(np.conj(profile)*other_profile)
        self.amplitude = projection/length_squared
        return self

    def profile(self, z=0, include_grid=False):
        ''' 
        Builds the profile of the mode along the optical axis.
        ---------------
        Parameters:
            z: float, optional
                Position along the optical axis [m].
                Defaults to 0.
            include_grid: bool, optional
                If True, the meshgrid of the positions is returned.
                Defaults to False.
        ---------------
        Returns:
            if include_grid:
                X, Y, profile
            else:
                profile
        ---------------
        '''
        w = self.width(z)
        X,Y = np.meshgrid(self.x, self.y)
        R = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y,X)
        l = self.order[1]
        p = self.order[0]
        N = 2*p + np.abs(l)
        rscale = (np.sqrt(2)*R/w)**np.abs(l)
        Lpl = self.polynomial(2*R**2/w**2, p, np.abs(l))
        expPhi = np.exp(-1j*l*phi)
        gaussian = self.gaussian_profile(z, N=N)
        output = rscale*Lpl*expPhi*gaussian
        return (X,Y,output) if include_grid else output


class Transform():

    def __init__(self, spectrum, kind='hermite', *args, **kwargs):
        '''
        Class that performs a transform with respect to a set of transverse modes.
        -----------------
        Parameters:
            spectrum: int or tuple of int
                Up to which transverse orders the transform needs to be computed.
                If int, a square grid is assumed.
            kind: str, optional
                Type of transverse transform. 
                Currently, 'hermite' and 'laguerre' are available.
                Defaults to 'hermite'.
            args: arguments for GaussianBeam
            kwargs: keyword-arguments for GaussianBeam
        -----------------
        '''
        self.kind = kind

        if self.kind == 'hermite':
            Mode = HermiteMode
        elif self.kind == 'laguerre':
            Mode = LaguerreMode
        else:
            complain = 'Selected kind not available. '
            suggestion = 'Please select either ''hermite'' or ''laguerre''.'
            raise Exception(complain + suggestion)

        region = kwargs['region']
        resolution = kwargs['resolution']
        if type(resolution) is not tuple:
            resolution = (resolution, resolution)
        if len(region) == 2:
            region = (region[0], region[1], region[0], region[1])
        if type(spectrum) is not tuple:
            spectrum = (spectrum, spectrum)
        if 'order' in kwargs:
            kwargs.pop('order', None)
        self.region = kwargs['region']
        self.resolution = kwargs['resolution']
        self.spectrum = spectrum
        self.modes = [[] for i in range(self.spectrum[0])]
        for v in range(spectrum[0]):
            for h in range(spectrum[1]):
                self.modes[v].append(Mode(order=(v,h), **kwargs))
        self.coef = None
        self.power_coef = None
        self.transformed = False
        self.power_transformed = False
        self.x = np.linspace(region[0], region[1], resolution[0])
        self.y = np.linspace(region[2], region[3], resolution[1])
        self.reconstruction = None

    def grid(self):
        '''
        ----------------
        Returns: 
            tuple of numpy.ndarray: the grid over which the profiles are evaluated.
        ----------------
        '''
        return np.meshgrid(self.x, self.y)

    def transform(self, profile, z=0):
        '''
        Decomposes a given profile in a superposition of transverse modes of kind self.kind.
        ---------------
        Parameters:
            profile: numpy.ndarray of dtype = 'complex'
                a two dimensional grid of complex electric field over the region of space
                of the modes.
            z: float, optional
                position along the optical axis [m] where the profile has been computed.
                Defaults to 0.
        ----------------
        Returns: 
            Instance of Transform
        ----------------
        '''
        self.reconstruction = None
        self.transformed = False
        self.coef = np.zeros((self.spectrum[0], self.spectrum[1]), dtype='complex')
        for h in tqdm(range(self.spectrum[1])):
            for v in range(self.spectrum[0]):
                self.coef[v,h] = self.modes[v][h].project(profile).amplitude
        self.transformed = True
        self.reconstruction = profile
        return self

    def propagate(self, z, include_grid=False):
        '''
        Propagation by separate propagation of the transverse modes.
        ---------------
        Parameters:
            z: float
                Position along the optical axis [m].
            include_grid: bool, optional
                If True, the meshgrid of positions is returned.
                Defaults to False.
        ---------------
        profile: numpy.ndarray of dtype = 'Complex'
                 complex electric field over the space grid.
        Returns:
            if include_grid:
                X, Y, profile
            else:
                profile
        --------------
        '''
        if not self.transformed:
            raise Exception('Transform has not been executed.')
        X,Y = self.grid()
        output = np.zeros_like(X, dtype='complex')
        flip_power_transformed = not self.power_transformed
        if not self.power_transformed:
            self.power_coef = np.zeros((self.spectrum[0], self.spectrum[1]))
        for h in tqdm(range(self.spectrum[0])):
            for v in range(self.spectrum[1]):
                mode_profile = self.modes[v][h].profile(z=z)
                output += mode_profile
                if not self.power_transformed:
                    mode_power = np.sum(np.abs(mode_profile)**2)
                    self.power_coef[v,h] = mode_power
        self.power_coef /= np.sum(np.abs(output)**2)
        if flip_power_transformed:
            self.power_transformed = not self.power_transformed
        return (X,Y,output) if include_grid else output

    def power_fraction(self):
        '''
        Given the amplitudes of the transverse transform,
        calculates the power contained by each transverse mode.
        ---------------
        Returns:
            numpy.ndarray: two-dimensional grid of floats containing the 
                           fraction of the power contained in each of the
                           corresponding transverse modes.
        --------------
        '''
        self.power_transformed = False
        if self.reconstruction is None:
            raise Exception('No transform has been performed.')
        total_power = np.sum(np.abs(self.reconstruction)**2)
        self.power_coef = np.zeros((self.spectrum[0], self.spectrum[1]))
        for h in tqdm(range(self.spectrum[1])):
            for v in range(self.spectrum[0]):
                mode_profile = self.modes[v][h].profile()
                mode_power = np.sum(np.abs(mode_profile)**2)
                self.power_coef[v,h] = mode_power/total_power
        self.power_transformed = True
        return self
            

