import numpy as np

# states
h = np.array([[1],[0]]) # horizontal
v = np.array([[0],[1]]) # vertical
d = 1/np.sqrt(2)*(h+v) # diagonal
a = 1/np.sqrt(2)*(h-v) # anti-diagonal
r = 1/np.sqrt(2)*(h-1j*v) # right handed circular
l = 1/np.sqrt(2)*(h+1j*v) # left handed circular

# polarizers
HP = np.array([[1,0],[0,0]]) # linear horizontal polarizer
VP = np.array([[0,0],[0,1]]) # linear vertical polarizer
RP = 1/2*np.array([[1,1j],[-1j,1]]) # circular right polarizer
LP = 1/2*np.array([[1,-1j],[1j,-1]]) # circular left polarizer

# phase retarders
QW = np.exp(-1j*np.pi/4)*np.array([[1,0],[0,1j]]) # quarter-wave plate with fast axis horizontal
HW = np.exp(-1j*np.pi/2)*np.array([[1,0],[0,-1]]) # half-wave plate with fast axis horizontal

def AR(eta):
    '''retards by arbitrary angle eta, fast axis is horizontal'''
    return np.array([[np.exp(-1j*eta/2),0],[0,np.exp(1j*eta/2)]])

# rotation
def rotation_matrix(theta):
    '''rotation matrix'''
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def rotate(v, theta):
    '''rotates matrix or vector by angle theta'''
    if v.shape[1] == 1:
        return rotation_matrix(theta) @ v
    elif v.shape[1] == 2:
        return rotation_matrix(theta) @ v @ rotation_matrix(-theta)

def power(state):
    '''gives the to 1 normalized power of intensity of the state'''
    return abs(np.vdot(state, state))
    # return norm(state)

def overlap(state1, state2):
    '''gives overlap between state1 and state2'''
    return abs(np.vdot(state1, state2))
