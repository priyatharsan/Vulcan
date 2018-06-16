import tensorflow as tf

import numpy as np

import utils
from utils import get_one_hot
from net import Network

from vulcanai import mnist_loader

from sklearn.metrics import accuracy_score

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

test_labels = get_one_hot(test_labels)
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28))

tf.reset_default_graph()
with tf.Graph().as_default() as tf_graph:
    input_placeholder_1, y = utils.initialize_pl(tf.float32, tf.float32,
                                    [None, 28, 28, 1],
                                    [None, 10])

test_images = np.expand_dims(test_images, axis=3)
size = 5000  # Chsnge the size as much needed based on the test
test_x, test_y = test_images[:size], test_labels[:size]
#pred_y = []
with tf.Session(graph=tf_graph) as sess:
    load_path = 'models/2018_06_13_173841_1_dense/'
    model_name = '1_dense_model'

    saver = Network.load_model(sess, load_path, model_name)
    init = tf.global_variables_initializer()
    pred = sess.run('dense_net/classification_layer/dense/Softmax:0',
                    feed_dict={
                                'input_placeholder_1:0': test_x})

    loss_var = sess.run(tf.get_collection(tf.GraphKeys.VARIABLES, scope='loss'))
    print loss_var

correct_pred = accuracy_score([[np.argmax(i)] for i in test_y],
                                    [[np.argmax(pred_y)] for pred_y in pred])
print("Accuracy = {}%".format(correct_pred*100))
