# Python 3 version - PyTorch Implementation


from datetime import datetime

import numpy as np

from sklearn.preprocessing import LabelBinarizer


def get_timestamp():
    """Return a 14 digit timestamp."""
    return datetime.now().strftime('%Y_%m_%d_%H%M%S_')


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Args:
        in_matrix: the categorized 1D matrix

    Returns: a one-hot matrix representing the categorized matrix
    """
    if in_matrix.dtype.name == 'category':
        custum_array = in_matrix.cat.codes

    elif isinstance(in_matrix, np.ndarray):
        custum_array = in_matrix

    else:
        raise ValueError("Input matrix cannot be converted.")

    lb = LabelBinarizer()
    return np.array(lb.fit_transform(custum_array), dtype='float32')
