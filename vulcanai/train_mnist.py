
from net import Network
from utils import get_one_hot
from vulcanai import mnist_loader

import tensorflow as tf
import utils

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

train_labels = get_one_hot(train_labels)


network_dense_config = {
    'mode': 'dense',
    'units': [512, 100],
    'dropouts': [0.2, .05],
}

with tf.Graph().as_default() as tf_graph:
    input_var, y = utils.initialize_pl(tf.float32, tf.float32,
                                      [None,len(train_images[0])]
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
    optimizer='adam')

with tf.Session(graph=tf_graph) as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('./tf_logs', sess.graph)
    sess.run(init)

    dense_net.train(sess,
        epochs=200,
        train_x=train_images[:50000],
        train_y=train_labels[:50000],
        val_x=train_images[50000:60000],
        val_y=train_labels[50000:60000],
        summary_writer =summary_writer,
        saver=saver,
        batch_ratio=0.05,
        plot=True
    )
