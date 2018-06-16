from sklearn.utils import shuffle

import numpy as np
import os

from net import Network
from utils import get_one_hot
from vulcanai import mnist_loader

import utils

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

train_images, train_labels = shuffle(train_images, train_labels, random_state=0)

# f_net = Network.load_model('models/20170828235548_fashion.network')
# m_net = Network.load_model('models/20170828235251_mnist.network')
# f_max_net = Network.load_model('models/20170902174725_1_dense_max.network')

label_map = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}


train_labels = get_one_hot(train_labels)
test_labels = get_one_hot(test_labels)

train_images = np.reshape(train_images, (train_images.shape[0], 28, 28))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28))

with tf.Graph().as_default() as tf_graph:
    input_var, y = utils.initialize_pl(tf.float32, tf.float32,
                                        [None, 28, 28, 1],
                                        [None, 10])
    network_conv_config = {
        'mode': 'conv',
        'filters': [16, 32],
        'filter_size': [[5, 5], [5, 5]],
        'stride': [[1, 1], [1, 1]],
        'pool': {
            'mode': 'MAX',  # 'MAX' or 'AVG'
            'stride': [[2, 2], [2, 2]]
        }
    }

    network_dense_config = {
        'mode': 'dense',
        'units': [512],
        'dropouts': [0.3],
    }
    with tf.variable_scope("conv_net"):
        conv_net = Network(
            name='conv_test',
            dimensions=input_var.shape,
            input_var=input_var,
            y=y,
            config=network_conv_config,
            input_network=None,
            num_classes=None
            )

    with tf.variable_scope("dense_net"):
        dense_net = Network(
            name='1_dense',
            dimensions=(None, int(train_images.shape[1])),
            input_var=input_var,
            y=y,
            config=network_dense_config,
            input_network={'network': conv_net, 'layer': 4, 'get_params': True},
            num_classes=10,
            activation='rectify',
            pred_activation='softmax'
            )

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

gpu_options = tf.GPUOptions(allow_growth=True)
log_path = 'tf_logs'
train_x, train_y = train_images[:5000], train_labels[:5000]
val_x, val_y = train_images[5000:6000], train_labels[5000:6000]
print("Size of:")
print("\t Training-set:\t\t{} {}".format(len(train_x), len(train_y)))
print("\t Validation-set:\t{} {}".format(len(val_x), len(val_y)))


# Instantiating Training.....
with tf.Session(graph=tf_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    dense_net.train(sess,
        epochs=2,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        log_path =log_path,
        batch_ratio=0.005,
        plot=False
    )
    inputs = {
            input_var.name: input_var,
            y.name: y,
        }
    outputs = {dense_net.network.name: dense_net.network}
    save_path = os.path.join('models2/', "{}".format(utils.get_timestamp()))
    dense_net.save_model2(sess, inputs, outputs, save_path)
    values = []
    for i in range(2):
        values.append(sess.run(dense_net.accuracy, {
            input_var: train_x,
            y: train_y
            }))


with tf.Session(graph=tf_graph) as sess2:
    # Restore saved values
    print('\nRestoring...')
    tf.saved_model.loader.load(
        sess2,
        [tag_constants.SERVING],
        save_path
    )
    print('Ok')
    # Get restored placeholders
    input_var2 = tf_graph.get_tensor_by_name(input_var.name)
    y2 = tf_graph.get_tensor_by_name(y.name)
    # Get restored model output
    network2 = tf_graph.get_tensor_by_name(dense_net.accuracy.name)
    restored_values=[]
    for i in range(2):
        restored_values.append(sess2.run(network2, {
            input_var2: train_x,
            y2: train_y
            }))
    print('Values: {}'.format(values))
    print('Restored values: {}'.format(restored_values))

# Check if original inference and restored inference are equal
valid = all((v == rv).all() for v, rv in zip(values, restored_values))
print('\nInferences match: {}'.format(valid))
