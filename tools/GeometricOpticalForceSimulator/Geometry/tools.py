import numpy as np


def is_number(candidate):
    """
    Returns true if the given value is a number
    :param candidate: value
    :return: true or false
    """
    compare_type = type(candidate)
    return compare_type == int or compare_type == float or compare_type == np.int or compare_type == np.float or compare_type == np.float16 or compare_type == np.float32 or compare_type == np.float64