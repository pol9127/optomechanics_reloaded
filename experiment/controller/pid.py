from __future__ import division, print_function, unicode_literals


class PIDController(object):
    def __init__(self, k_p, k_i, k_d, set_point=0,
                 output_max=100, output_min=0, init_output=0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self._set_point = set_point
        self.last_error = 0
        self.output_max = output_max
        self.output_min = output_min
        self._integral = init_output/self.k_i

    @property
    def set_point(self):
        # type: () -> float
        return self._set_point

    @set_point.setter
    def set_point(self, new_set_point):
        # type: float
        self._set_point = new_set_point

        # initialize error
        self.last_error = 0

    @property
    def integral(self):
        # type: () -> float
        return self._integral

    @integral.setter
    def integral(self, value):
        self._integral = value

        integral_limits = [self.output_min/self.k_i,
                           self.output_max/self.k_i]

        if self._integral > max(integral_limits):
            self._integral = max(integral_limits)
        elif self._integral < min(integral_limits):
            self._integral = min(integral_limits)

    def update(self, current_value, time_step=1.0):
        error = self.set_point - current_value
        self.integral += error*time_step
        derivative = (error - self.last_error) / time_step

        output = (self.k_p*error + self.k_i*self.integral +
                  self.k_d*derivative)

        if output > self.output_max:
            output = self.output_max
        elif output < self.output_min:
            output = self.output_min

        return output
