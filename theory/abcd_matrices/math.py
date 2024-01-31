import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib

def refractive_index_to_color(n, n_range=[1, 5]):
    if n > n_range[1]:
        n = n_range[1]
    elif n < n_range[0]:
        n = n_range[0]
    gb = 255 - int((n - n_range[0]) / (n_range[1] - n_range[0]) * 255)
    return '#%02x%02x%02x' % (gb, gb, 255)
