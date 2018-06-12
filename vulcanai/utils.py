import tensorflow as tf

from datetime import datetime

import numpy as np

from sklearn.preprocessing import LabelBinarizer


def initialize_pl(x_dtype, y_type, x_shape=None, y_shape=None):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    """

    X_placeholder = tf.placeholder(x_dtype, shape=(x_shape),
                                  name = 'input_placeholder')
    Y_placeholder = tf.placeholder(y_type, shape=(y_shape),
                                  name = 'labels_placeholder')

    return X_placeholder, Y_placeholder


def get_timestamp():
    """Return a 14 digit timestamp."""
    return datetime.now().strftime('%Y%m%d%H%M%S_')


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
