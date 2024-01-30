import optomechanics.experiment.lab_control.devices.piezo_stage._smaract as _smaract
import numpy as np
from copy import copy
from time import sleep

class _SmarAct(object):
    _configuration = {'Acquisition': {'SampleRate': 1.,
                                      'MaxSampleSize': 10000}}
    def __init__(self, location=None):
        if location is None:
            _smaract.init()
        else:
            _smaract.init(location)

    @property
    def configuration(self):
        self._configuration['Acquisition'].update(_smaract.get_configuration())
        return self._configuration

    @configuration.setter
    def configuration(self, config):
        for key in config['Acquisition']:
            self._configuration['Acquisition'][key] = config['Acquisition'][key]

    @property
    def channel_configuration(self):
        return _smaract.get_channel_configuration(0)

    def stickslip_mode(self, mode=0):
        mode = int(mode)
        return _smaract.set_channel_stickslip_mode(mode)

    def set_movement(self, acceleration=0, speed=0):
        _smaract.set_closed_loop_move_acceleration(acceleration)
        _smaract.set_closed_loop_move_speed(speed)

    @property
    def position(self):
        return _smaract.get_position()

    def move_relative(self, ch, pos):
        _smaract.move_relative(ch, pos)

    def move_absolute(self, ch, pos):
        _smaract.move_absolute(ch, pos)

    @staticmethod
    def meander3d(Dx, Dy, Dz, nx, ny, nz):
        x = np.linspace(Dx[0], Dx[1], nx, dtype=int)
        y = np.linspace(Dy[0], Dy[1], ny, dtype=int)
        z = np.linspace(Dz[0], Dz[1], nz, dtype=int)
        X, Y, Z = np.meshgrid(x, y, z)

        order = []
        for _z in np.arange(len(z)):
            orderXY = np.vstack((np.arange(nx) + _y * nx for _y in np.arange(ny)))
            orderXY[1::2, :] = orderXY[1::2, ::-1]
            if _z % 2 == 0:
                order.append(orderXY + _z * nx * ny)
            else:
                order.append((orderXY + _z * nx * ny).flatten()[::-1].reshape(nx, ny))
        order = np.array(order)

        dx = copy(X)
        dy = copy(Y)
        dz = copy(Z)
        for i in np.arange(nx * ny * nz - 1):
            dx[order == i + 1] = X[order == i + 1] - X[order == i]
            dy[order == i + 1] = Y[order == i + 1] - Y[order == i]
            dz[order == i + 1] = Z[order == i + 1] - Z[order == i]

        return [dx.flatten()[order.flatten()], dy.flatten()[order.flatten()], dz.flatten()[order.flatten()]], [X, Y, Z], order

    @property
    def status(self):
        return _smaract.get_status()




    def __del__(self):
        try:
            _smaract.close()
        except:
            pass


if __name__ == '__main__':
    from time import sleep
    from timeit import default_timer
    import matplotlib.pyplot as plt
    piezo = _SmarAct('usb:id:459394397')
    dz = 200    # nm
    # print(piezo.status)
    # piezo.move_relative(2, dz)
    #
    # sleep(3)
    # print(piezo.position)
    # piezo.move_relative(2, -dz)


    ### Check position poll rate:
    # N = 1000
    # m = 5
    # times = []
    # for _ in range(m):
    #     t0 = default_timer()
    #     for i in range(N):
    #         piezo.position
    #     times.append(default_timer() - t0)
    # print(times, np.mean(times))

    ### Check movement speed
    # posi = [piezo.position[2]]
    # times = [0]
    # N = 300
    # t0 = default_timer()
    # for i in range(N):
    #     if i == 20:
    #         piezo.move_relative(2, dz)
    #     elif posi[-1] > -4200:
    #         print(default_timer() - t0)
    #         piezo.move_relative(2, -dz)
    #     else:
    #         posi.append(piezo.position[2])
    #         times.append(default_timer() - t0)
    # fig, ax = plt.subplots()
    # ax.plot(times, posi)
    # ax.grid(True)
    # plt.show()
    print(piezo.position)
    # piezo.move_relative(2, 500)