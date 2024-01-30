from optomechanics.experiment.device.laser.NKT import NKTP_DLL as nkt

class Module:
    devID = None
    port = None
    open_port = True

    def __init__(self, port, devID, open_port=True):
        self.devID = devID
        self.port = port
        self.open_port = open_port
        if self.open_port:
            openResult = nkt.openPorts(self.port, 0, 0)
            print('Opening the comport:', nkt.PortResultTypes(openResult))

    def __del__(self):
        if self.open_port:
            closeResult = nkt.closePorts(self.port)
            print('Close the comport:', nkt.PortResultTypes(closeResult))

    @property
    def status(self):
        rdResult, stat = nkt.deviceGetStatusBits(self.port, self.devID)
        return stat


class Module_20(Module):
    @property
    def emission(self):
        rdResult, emission_state = nkt.registerReadU8(self.port, self.devID, 0x30, -1)
        return emission_state

    @emission.setter
    def emission(self, state):
        if not isinstance(state, bool):
            try:
                state = bool(state)
            except:
                print('Emission state must be 1 (On) or 0 (Off)')
        wrResult = nkt.registerWriteU8(self.port, self.devID, 0x30, int(state), -1)

    @property
    def wavelength_offset(self):
        rdResult, wavelength = nkt.registerReadU16(self.port, self.devID, 0x28, -1)
        return wavelength

    @wavelength_offset.setter
    def wavelength_offset(self, new_wavelength):
        wrResult = nkt.registerWriteU16(self.port, self.devID, 0x28, new_wavelength, -1)

    @property
    def power(self):
        rdResult, pow = nkt.registerReadU16(self.port, self.devID, 0x23, -1)
        return pow

    @power.setter
    def power(self, new_power):
        wrResult = nkt.registerWriteU16(self.port, self.devID, 0x23, new_power, -1)

    @property
    def wavelength(self):
        rdResult, wavelength = nkt.registerReadU16(self.port, self.devID, 0x25, -1)
        return wavelength


class Module_33(Module):
    @property
    def wavelength(self):
        rdResult, wavelength = nkt.registerReadU32(self.port, self.devID, 0x32, -1)
        return wavelength

    @property
    def wavelength_actual(self):
        rdResult, wavelength = nkt.registerReadS32(self.port, self.devID, 0x72, -1)
        return wavelength

class Module_34(Module):
    @property
    def emission(self):
        rdResult, emission_state = nkt.registerReadU8(self.port, self.devID, 0x30, -1)
        return emission_state

    @emission.setter
    def emission(self, state):
        if not isinstance(state, bool):
            try:
                state = bool(state)
            except:
                print('Emission state must be 1 (On) or 0 (Off)')
        wrResult = nkt.registerWriteU8(self.port, self.devID, 0x30, int(state), -1)

    @property
    def wavelength_offset(self):
        rdResult, wavelength = nkt.registerReadS16(self.port, self.devID, 0x2D, -1)
        return wavelength

    @wavelength_offset.setter
    def wavelength_offset(self, new_wavelength):
        wrResult = nkt.registerWriteS16(self.port, self.devID, 0x2D, new_wavelength, -1)

    @property
    def power(self):
        rdResult, pow = nkt.registerReadU16(self.port, self.devID, 0x2F, -1)
        return pow

    @power.setter
    def power(self, new_power):
        wrResult = nkt.registerWriteU16(self.port, self.devID, 0x2F, new_power, -1)


class Adjustik_K822:
    basik = None
    adjustik = None
    def __init__(self, port, basikID=1, adjustikID=128):
        self.basik = Module_33(port, basikID)
        self.adjustik = Module_34(port, adjustikID, open_port=False)

    @property
    def emission(self):
        return self.adjustik.emission

    @emission.setter
    def emission(self, state):
        self.adjustik.emission = state

    @property
    def wavelength(self):
        # Return the wavelength in nm.
        return self.basik.wavelength * 1e-4 + self.adjustik.wavelength_offset * 1e-4

    @wavelength.setter
    def wavelength(self, new_wavelength):
        # Set the wavelength. The new wavelength is expected to be given in nm
        new_wavelength_offset = int(new_wavelength * 1e4 - self.basik.wavelength)
        self.adjustik.wavelength_offset = new_wavelength_offset

    @property
    def status(self):
        return self.basik.status, self.adjustik.status

    @property
    def wavelength_stable(self):
        # returns 1 if wavelength is stable
        if self.basik.status > 15:
            return 0
        else:
            return 1

    @property
    def power(self):
        # returns power in mW
        return self.adjustik.power * 1e-2

    @power.setter
    def power(self, new_power):
        self.adjustik.power = int(new_power * 1e2)

    @property
    def wavelength_actual(self):
        return self.basik.wavelength_actual * 1e-4 + self.basik.wavelength * 1e-4

if __name__ == '__main__':
    port = 'COM5'
    al = Adjustik_K822(port)
    print(al.wavelength_actual)
    al.emission = 0
    al.power = 12
    al.wavelength = 1550.2300
    print('Wavelength Stable:', al.wavelength_stable)
    print('Emission:', al.emission)
    print('Power:', al.power)
    print('Wavelength:', al.wavelength)



