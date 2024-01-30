import matplotlib.pyplot as plt
import numpy as np

def cyclic_cmap(c1, c2, c3=[255, 255, 255], cmap_name='custom_cmap'):
    """Create a cyclic color map. The colour range will go from c1 to c3 to c2 to c3 to c1. The colour will be
    registered as a cmap with the passed name.
    Note
    ----
    If no third colour is specified it will use white instead.

    Parameters
    ----------
    c1
        list of:
        -   rgb values for the first colour (Exp: [255, 0, 0] for red)
    c2
        list of:
        -   rgb values for the first colour (Exp: [255, 0, 0] for red)
    c3
        list of:
        -   rgb values for the first colour (Exp: [255, 0, 0] for red)
    cmap_name
        string:
        -   name of the new colormap

    Returns
    -------
    Nothing, only registers cmap
    """

    if not isinstance(c1, np.ndarray):
        c1 = np.array(c1, dtype=float)
    if not isinstance(c2, np.ndarray):
        c2 = np.array(c2, dtype=float)
    if not isinstance(c3, np.ndarray):
        c3 = np.array(c3, dtype=float)

    c1 /= 255
    c2 /= 255
    c3 /= 255

    colors = ['red', 'green', 'blue']
    cdict = {cl: ((0.00, c1[i], c1[i]),
                  (0.25, c3[i], c3[i]),
                  (0.50, c2[i], c2[i]),
                  (0.75, c3[i], c3[i]),
                  (1.00, c1[i], c1[i])) for cl, i in zip(colors, range(3))}

    plt.register_cmap(name=cmap_name, data=cdict)

cyclic_1 = cyclic_cmap([0, 51, 102], [255, 255, 0], [0, 153, 153], 'cyclic_1')