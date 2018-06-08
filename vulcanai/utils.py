import tensorflow as tf

from datetime import datetime

import numpy as np

from sklearn.preprocessing import LabelBinarizer


def init_placeholders(feature_size, num_labels, batch_size = None):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    """

    X_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            feature_size]),
                                                            name = 'input')
    Y_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            num_labels]),
                                                            name = 'truth')

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
