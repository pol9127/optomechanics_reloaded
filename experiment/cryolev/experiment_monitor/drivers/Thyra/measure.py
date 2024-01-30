import serial
import numpy as np
import time

def get_cs(st):
    num = np.mod(np.sum(np.array(list(st))), 64)+64
    charac = chr(num)
    return charac


def measure(name, verbose = False):
    st = b'0010MV00'
    charac = get_cs(st)
    st1 = str(st)[1:][1:-1] + charac + '\r'
    b_str = bytes(st1, 'utf-8')
    if verbose:
        print("connect to serial")
    ser = serial.Serial(name, 115200, timeout = .1)#for example, name = '/dev/ttyUSB0'.
    ser.write(b_str)
    #ser._timeout = 3
    if verbose:
        print("serial reading")
    st2 = str(ser.read_until('\r', 18))
    if st2.endswith("\\r'"):
        if st2[6:8] == 'MV':
            if st2[2:6] == "0011":
                try:
                    if verbose:
                        print(st2)
                    ind = 0
                    for i in st2:
                        if i == 'V':
                            ind = st2.index(i)
                    length = int(float(st2[ind+1:ind+3]))
                    res = st2[ind+3:ind+3+length]
                    res = float(res)
                    if verbose:
                        print("got reasult from serial")
                    return res
                except:
                    ser.reset_input_buffer()
                    print('The recevied bytestring does not match the expected format\n')
                    return None
            else:
                ser.reset_input_buffer()
                print('The recevied bytestring does not match the expected format\n')
                return None
        else:
            ser.reset_input_buffer()
            print('The recevied bytestring does not match the expected format\n')
            return None
    else:
        ser.reset_input_buffer()
        print('The recevied bytestring does not match the expected format\n')
        return None
