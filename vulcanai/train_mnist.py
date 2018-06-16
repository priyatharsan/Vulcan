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

train_labels = get_one_hot(train_labels)


network_dense_config = {
    'mode': 'dense',
    'units': [512],
    'dropouts': [0.2],
}


log_path = 'tf_logs'
train_x, train_y = train_images[:50000], train_labels[:50000]
val_x, val_y = train_images[50000:60000], train_labels[50000:60000]
print("Size of:")
print("\t Training-set:\t\t{} {}".format(len(train_x), len(train_y)))
print("\t Validation-set:\t{} {}".format(len(val_x), len(val_y)))


with tf.Graph().as_default() as tf_graph:
    input_var, y = utils.initialize_pl(tf.float32, tf.float32,
                                      [None,len(train_images[0])],
                                      [None,len(train_labels[0])])
    dense_net = Network(
                name='3_dense_test',
                dimensions=[None] + list(train_images.shape[1:]),
                input_var=input_var,
                y=y,
                config=network_dense_config,
                input_network=None,
                num_classes=10,
                activation='rectify',
                pred_activation='softmax',
                optimizer='adam'
                )

with tf.Session(graph=tf_graph) as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    sess.run(init)

    dense_net.train(sess,
        epochs=2,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        log_path =log_path,
        batch_ratio=0.05,
        plot=False
    )
    #dense_net.save_model(sess, saver)
    inputs = {
            input_var.name: input_var,
            y.name: y,
        }
    outputs = {dense_net.network.name: dense_net.network}
    save_path = os.path.join('models2/', "{}".format(utils.get_timestamp()))
    dense_net.save_model2(sess, inputs, outputs, save_path)
    values = []
    for i in range(2):
        values.append(sess.run(dense_net.network, {
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
    network2 = tf_graph.get_tensor_by_name(dense_net.network.name)
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
