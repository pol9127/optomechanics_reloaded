"""
@author: Andrei Militaru
@date: 22nd September 2020
"""

import numpy as np

class Orthonormalizer:
    
    def __init__(self, *args, check=False):

        vectors = []
        for vector in args:
            vectors.append(vector)
        vectors = np.array(vectors)

        if check and (np.linalg.det(vectors) == 0):
            raise Exception('The vectors are not linearly independent.')

        self._vectors = np.array(vectors)
        self._checked = False
        self._orthonormalized = False
        self._ortho_vectors = None
        self.matrix = None

    @property
    def orthonormalized(self):
        if self._checked:
            return self._orthonormalized
        else:
            self._checked = False
            accumulate = 0
            n, m = self.vectors.shape
            for i in range(n):
                for j in range(i + 1, n):
                    print(self.vectors[i], self.vectors[j], np.dot(self.vectors[i], self.vectors[j]))
                    accumulate += np.dot(self.vectors[i], self.vectors[j])
            self._checked = True
            self._orthonormalized = True if accumulate == 0 else False
            return self._orthonormalized

    @property
    def vectors(self):
            return self._vectors

    @property
    def ortho_vectors(self):
        return self._ortho_vectors

    def basis_change(self):
        """from orthonormal to non-orthonormal"""

        coefficients = []
        for i in range(len(self.vectors)):
            coefficients.append(self.project(self.vectors[i], basis='ortho'))
        self.matrix = np.array(coefficients)
        return self

    def orthonormalize(self):
        new_vectors = [self.vectors[0]/np.sqrt(np.dot(self.vectors[0], self.vectors[0]))]
        n, m = self.vectors.shape
        for i in range(1, n):
            vector = 1*self.vectors[i]
            correction = np.zeros(m)
            for j in range(i):
                correction += new_vectors[j]*np.dot(self.vectors[i], new_vectors[j])
            vector -= correction
            new_vectors.append(vector/np.sqrt(np.dot(vector, vector)))
        self._ortho_vectors = np.array(new_vectors)
        return self

    def generate(self, *coefficients):
        if len(coefficients) != len(self.vectors):
            raise Exception('Number of coefficients needs to match the dimension of the space.')
        else:
            output = np.zeros(len(self.vectors[0]))
            for (coeff, vector) in zip(coefficients, self.vectors):
                output += coeff*vector
            return output

    def project(self, vector, basis='ortho'):
        if self._ortho_vectors is None:
            self.orthonormalize()
        vector = np.array(vector)
        coefficients = []
        for i in range(len(self.vectors)):
            coefficients.append(np.dot(vector, self.ortho_vectors[i]))
        if basis == 'ortho':
            return np.array(coefficients)
        elif basis == 'original':
            if self.matrix is None:
                self.basis_change()
            return np.array(coefficients) @ np.linalg.inv(self.matrix)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    time = np.linspace(0, 10, 10000)
    vec1 = np.ones_like(time)
    vec2 = np.exp(-time)
    vec3 = np.exp(-2*time)
    vec4 = np.exp(-3*time)
    vec5 = np.exp(-4*time)
    vec6 = np.exp(-5*time)
    vec7 = np.exp(-6*time)
    basis = Orthonormalizer(vec1, vec2, vec3, vec4, vec5, vec6, vec7)
    (ovec1, ovec2, ovec3, ovec4, ovec5, ovec6, ovec7) = basis.orthonormalize().ortho_vectors
    """
    print(np.dot(ovec7, ovec5), np.dot(ovec5, ovec5), np.dot(ovec7, ovec7))
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(time, vec1, time, vec2, time, vec3, time, vec4, time, vec5, time, vec6, time, vec7)
    ax2.plot(time, ovec1, time, ovec2, time, ovec3, time, ovec4, time, ovec5, time, ovec6, time, ovec7)
    plt.show()"""

    probe = basis.generate(1, 3, 0, 0.5, 2, 1.5, 3)
    probe_noise = probe + np.random.normal(size=len(probe))

    def fitting(t, a, b, c, d, e, f, g):
        return a + b*np.exp(-t) + c*np.exp(-2*t) + d*np.exp(-3*t) + d*np.exp(-3*t) + e*np.exp(-4*t) + f*np.exp(-5*t) + g*np.exp(-6*t)

    params, cov = curve_fit(fitting, time, probe_noise, [1.5, 3, 0, 0.5, 2, 1.5, 3])

    coeffs = basis.project(probe_noise, basis='original')
    #print(coeffs)
    #print(params)

    plt.plot(time, probe_noise, '.', label='noisy original')
    plt.plot(time, probe, label='original')
    plt.plot(time, basis.generate(*coeffs), '--', label='mode decomposition')
    plt.plot(time, basis.generate(*params), label='gradient descent')
    plt.legend()
    plt.show()
