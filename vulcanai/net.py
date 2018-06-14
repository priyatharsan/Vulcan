import time

import sys

import os

import numpy as np

import tensorflow as tf

from ops import activations, optimizers

from utils import get_timestamp

import json

import cPickle as pickle

import matplotlib.pyplot as plt


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, config,
                 input_network=None, num_classes=None, activation='rectify',
                 pred_activation='softmax', optimizer='adam', stopping_rule='best_validation_error',
                 learning_rate=0.001):

        """
        Initialize network specified.

        Args:
            name: string of network name
            dimensions: the size of the input data matrix
            input_var: tensor representing input matrix
            y: tensor representing truth matrix
            config: Network configuration (as dict)
            input_network: None or a dictionary containing keys (network, layer).
                network: a Network object
                layer: an integer corresponding to the layer you want output
            num_classes: None or int. how many classes to predict
            activation:  activation function for hidden layers
            pred_activation: the classifying layer activation
            optimizer: which optimizer to use as the learning function
            learning_rate: the initial learning rate
        """
        self.name = name
        self.input_dim = dimensions
        self.input_var = input_var
        self.y = y
        self.config = config

        self.num_classes = num_classes
        if not optimizers.get(optimizer, False):
            raise ValueError(
                'Invalid optimizer option: {}. '
                'Please choose from:'
                '{}'.format(optimizer, optimizers.keys()))
        if not activations.get(activation, False) or \
           not activations.get(pred_activation, False):
            raise ValueError(
                'Invalid activation option: {} and {}. '
                'Please choose from:'
                '{}'.format(activation, pred_activation, activations.keys()))
        self.activation = activation
        self.pred_activation = pred_activation
        self.optimizer = optimizer

        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [],
                                                    name='learning_rate')
        self.stopping_rule = stopping_rule

        self.input_network = input_network

        self.layers = []
        self.cost = None
        self.val_cost = None
        self.input_params = None
        if self.input_network is not None:
            if self.input_network.get('network', False) is not False and \
               self.input_network.get('layer', False) is not False and \
               self.input_network.get('get_params', None) is not None:
                self.layers = self.input_network['network'].layers
                self.input_var = self.layers[0].input
                self.input_dim = self.layers[self.input_network['layer']].shape

            else:
                raise ValueError(
                    'input_network for {} requires {{ network: type Network,'
                    ' layer: type int, get_params: type bool}}. '
                    'Only given keys: {}'.format(
                        self.name, self.input_network.keys()
                    )
                )

        if isinstance(self.input_dim, tf.TensorShape):
            self.input_dim = self.input_dim.as_list()

        self.network = self.create_network(
            config=self.config,
            nonlinearity=activations[self.activation])

        if self.num_classes is not None:
            self.cost, self.opt = self.training()
            self.accuracy = self.evaluation(self.network, self.y)

        try:
            self.timestamp
        except AttributeError:
            self.timestamp = get_timestamp()
        self.minibatch_iteration = 0

    def training(self):
        """
        Creates an optimizer and applies the gradients to all trainable variables.
        The train_op returned by this function is passed to the
        `sess.run()` call to cause the model to train.

        Creates a summarizer to track the loss over time in TensorBoard.

        Returns:
            train_step: The Op for training.
        """

        print("Calculating Loss")
        with tf.variable_scope('loss'):
            if self.num_classes is None or self.num_classes == 0:
                loss = self.mse_loss(self.network, self.y)
            else:
                loss = self.cross_entropy_loss(self.network, self.y)

            loss = tf.reduce_mean(loss)
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)
        print("Creating {} Trainer...".format(self.name))

        if self.optimizer == 'adam':
            optimizer = optimizers[self.optimizer](
                            learning_rate=self.learning_rate_placeholder,
                            beta1=0.9,
                            beta2=0.999,
                            epsilon=1e-08
            )
        elif self.optimizer == 'sgd':
            optimizer = optimizers[self.optimizer](
                            learning_rate=self.learning_rate_placeholder
            )
        else:
            updates = None
            ValueError("No optimizer found")
        trainable = tf.trainable_variables()
        with tf.name_scope("train_op_{}".format(self.name)):
            # Variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss,
                                        global_step=global_step,
                                        var_list=trainable)
        return loss, train_op

    def evaluation(self, logits, labels_placeholder):
        """
        Records the accuracy of the network.

        Returns: A scalar float32 tensor with the number of examples
                 that were predicted correctly.
        """
        print("Creating {} Validator...".format(self.name))

        with tf.name_scope("accuracy"):
            # check how much error in prediction
            if self.val_cost is None:
                if self.num_classes is None or self.num_classes == 0:
                    self.val_cost = self.mse_loss(logits, labels_placeholder)
                    val_acc = tf.constant(0)
                else:
                    self.val_cost = self.cross_entropy_loss(logits, labels_placeholder)
                    # check the accuracy of the prediction
                    if self.num_classes > 1:
                        val_acc = tf.reduce_mean(tf.cast(tf.equal(
                                              tf.argmax(logits, axis=1),
                                              tf.argmax(labels_placeholder, axis=1)),
                                              tf.float32))
                    elif self.num_classes == 1:
                        val_acc = tf.reduce_mean(tf.cast(tf.equal(
                                              tf.round(logits), labels_placeholder),
                                              tf.float32))
            tf.summary.scalar("accuracy", val_acc)

        return val_acc

    def create_network(self, config, nonlinearity):

        import jsonschema
        import schemas
        mode = config.get('mode')
        if mode == 'dense':
            jsonschema.validate(config, schemas.dense_network)

            network = self.create_dense_network(
                units=config.get('units'),
                dropouts=config.get('dropouts'),
                nonlinearity=nonlinearity
            )
        elif mode == 'conv':
            jsonschema.validate(config, schemas.conv_network)

            network = self.create_conv_network(
                filters=config.get('filters'),
                filter_size=config.get('filter_size'),
                stride=config.get('stride'),
                pool_mode=config['pool'].get('mode'),
                pool_stride=config['pool'].get('stride'),
                nonlinearity=nonlinearity
            )
        else:
            raise ValueError('Mode {} not supported.'.format(mode))

        if self.num_classes is not None and self.num_classes != 0:
            network = self.create_classification_layer(
                network,
                num_classes=self.num_classes,
                nonlinearity=activations[self.pred_activation]
            )

        return network

    def create_conv_network(self, filters, filter_size, stride,
                            pool_mode, pool_stride, nonlinearity):
        """
        Create a convolutional network (1D, 2D, or 3D).

        Args:
            filters: list of int. number of kernels per layer
            filter_size: list of int list. size of kernels per layer
            stride: list of int list. stride of kernels
            pool_mode: string. pooling operation
            pool_stride: list of int list. down_scaling factor
            nonlinearity: string. nonlinearity to use for each layer

        Returns a conv network
        """
        conv_dim = len(filter_size[0])
        tf_pools = ['MAX', 'AVG']
        if not all(len(f) == conv_dim for f in filter_size):
            raise ValueError('Each tuple in filter_size {} must have a '
                             'length of {}'.format(filter_size, conv_dim))
        if not all(len(s) == conv_dim for s in stride):
            raise ValueError('Each tuple in stride {} must have a '
                             'length of {}'.format(stride, conv_dim))
        if not all(len(p) == conv_dim for p in pool_stride):
            raise ValueError('Each tuple in pool_stride {} must have a '
                             'length of {}'.format(pool_stride, conv_dim))
        if pool_mode not in tf_pools:
            raise ValueError('{} pooling does not exist. '
                             'Please use one of: {}'.format(pool_mode, tf_pools))

        print("\nCreating {} Network...".format(self.name))

        if self.input_network is None:
            with tf.variable_scope('Input_layer'):
                print('\tInput Layer:')
                input_layer = tf.keras.layers.InputLayer(
                                    input_shape=self.input_dim,
                                    input_tensor=self.input_var,
                                    name="{}_input"
                                    .format(self.name))
                network = input_layer.output
                print('\t\t{} {}'.format(network.shape, network.name))
                self.layers.append(input_layer)
        else:
            network = self.input_network['network']. \
                layers[self.input_network['layer']]

            print('Appending layer {} from {} to {}'.format(
                self.input_network['layer'],
                self.input_network['network'].name,
                self.name))

        if conv_dim == 1:
            conv_layer = tf.layers.conv1d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling1d
            else:
                pool = tf.layers.max_pooling1d
        elif conv_dim == 2:
            conv_layer = tf.layers.conv2d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling2d
            else:
                pool = tf.layers.max_pooling2d
        elif conv_dim == 3:
            conv_layer = tf.layers.conv2d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling1d
            else:
                pool = tf.layers.max_pooling1d
        else:
            pool = None   # Linter is stupid
            conv_layer = None
            ValueError("Convolution is only supported for one of the first three dimensions")

        print('\tHidden Layers:')
        with tf.variable_scope("Hidden_layers"):
            for i, (f, f_size, s, p_s) in enumerate(zip(filters,
                                                        filter_size,
                                                        stride,
                                                        pool_stride)):
                layer_name = "conv{}D_layer{}".format(conv_dim, i)
                with tf.variable_scope(layer_name):
                    network = conv_layer(
                        inputs=network,
                        filters=f,
                        kernel_size=f_size,
                        strides=s,
                        padding='same',
                        activation=nonlinearity
                        )
                    self.layers.append(network)
                    print('\t\t{} {}'.format(network.shape, network.name))
                    network = pool(
                        inputs=network,
                        pool_size=p_s,
                        strides=p_s,
                        padding='same'
                        )
                self.layers.append(network)
                print('\t\t{} {}'.format(network.shape, network.name))
        return network

    def create_dense_network(self, units, dropouts, nonlinearity):
        """
        Generate a fully connected network of dense layers.

        Args:
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Tensorflow.nn

        Returns: the output of the network (linked up to all the layers)
        """
        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("\nCreating {} Network...".format(self.name))
        if self.input_network is None:
            with tf.variable_scope('Input_layer'):
                print('\tInput Layer:')
                input_layer = tf.keras.layers.InputLayer(
                                    input_shape=self.input_dim,
                                    input_tensor=self.input_var,
                                    name="{}_input"
                                    .format(self.name))
                network = input_layer.output
                print('\t\t{} {}'.format(network.shape, network.name))
                self.layers.append(input_layer)
        else:

            network = self.input_network['network']. \
                layers[self.input_network['layer']]
            print('Appending layer {} from {} to {}'.format(
                self.input_network['layer'],
                self.input_network['network'].name,
                self.name))

        if nonlinearity.__name__ == 'selu':
            network = tf.layers.batch_normalization(
                        network,
                        training=(mode == tf.estimator.ModeKeys.TRAIN),
                        name="{}_batchnorm".format(self.name))

        if len(network.shape) > 2 and self.config.get('mode') == 'dense':
            network = tf.layers.flatten(network)

        print('\tHidden Layers:')
        with tf.variable_scope("Hidden_layers"):
            for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
                layer_name = 'dense_layer%s' % i
                with tf.variable_scope(layer_name):
                    if nonlinearity.__name__ == 'selu':
                        new_layer = tf.layers.dense(
                                            inputs=network,
                                            units=num_units,
                                            activation=nonlinearity,
                                            kernel_initializer=tf.initializers
                                            .random_normal(stddev=np.sqrt(
                                                1.0 / num_units)),
                                            bias_regularizer=tf.initializers
                                            .random_normal(stddev=0.0),
                                            name="dense_selu")

                        network = tf.contrib.nn.alpha_dropout(new_layer,
                                                            prob_dropout)
                        tf.summary.histogram(layer_name + '/selu', network)
                    else:
                        new_layer = tf.layers.dense(
                                            inputs=network,
                                            units=num_units,
                                            activation=nonlinearity,
                                            )  # By default TF assumes Glorot uniform initializer for weights and zero initializer for bias

                        network = tf.nn.dropout(new_layer,
                                                        prob_dropout)
                        tf.summary.histogram(layer_name+'/'+self.activation, network)
                self.layers.append(network)
                print('\t\t{} {}'.format(network.shape, network.name))

        return network

    def create_classification_layer(self, network, num_classes,
                                    nonlinearity):
        """
        Create a classification layer. Normally used as the last layer.

        Args:
            network: network you want to append a classification to
            num_classes: how many classes you want to predict
            nonlinearity: nonlinearity to use as a string (see DenseLayer)

        Returns: the classification layer appended to all previous layers
        """
        print('\tOutput/Classification Layer:')
        with tf.variable_scope("classification_layer"):
            network = tf.layers.dense(
                                inputs=network,
                                units=num_classes,
                                activation=tf.nn.softmax
                                )

        print('\t\t{} {}'.format(network.shape, network.name))
        self.layers.append(network)
        return network

    def cross_entropy_loss(self, prediction, y):
        """Generate a cross entropy loss function."""
        print("Using categorical cross entropy loss")
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=prediction,
                    labels=y
                    ))

    def mse_loss(self, prediction, y):
        """Generate mean squared error loss function."""
        print("Using Mean Squared error loss")
        return tf.losses.mean_squared_error(
                    predictions=prediction,
                    labels=y
                    )

    def fill_feed_dict(self, x, y, lr):
        feed_dict = {
                self.input_var: x,
                self.y: y,
                self.learning_rate_placeholder: lr
                }
        return feed_dict


    def train(self, sess, epochs, train_x, train_y, val_x, val_y, summary_writer, saver,
              batch_ratio=0.1, plot=True, change_rate=None):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            batch_ratio: the percent (0-1) of how much data a batch should have
            plot: If True, plot performance during training
            change_rate: a function that updates learning rate (takes an alpha, returns an alpha)'

        """
        print('\nTraining {} in progress...\n'.format(self.name))

        self.sess = sess
        self.saver = saver
        self.summary_writer = summary_writer
        self.run_metadata = tf.RunMetadata()
        self.summaries = tf.summary.merge_all()

        if batch_ratio > 1:
            batch_ratio = 1
        self.batch_ratio = float(batch_ratio)

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )

        if self.stopping_rule == 'best_validation_error':
            best_state = None
            best_epoch = None
            best_error = float('inf')

        elif self.stopping_rule == 'best_validation_accuracy':
            best_state = None
            best_epoch = None
            best_accuracy = 0.0

        output_shape = self.network.shape
        if output_shape[1:] != train_y.shape[1:]:
            raise ValueError(
                'Shape mismatch: non-batch dimensions don\'t match.'
                '\n\tNetwork output shape: {}'
                '\n\tLabel shape (train_y): {}'.format(
                    output_shape,
                    train_y.shape))
        train_x_shape = train_x.shape
        if train_x_shape[0] * batch_ratio < 1.0:
            batch_ratio = 1.0 / train_x.shape[0]
            print('Warning: Batch ratio too small. Changing to {:.5f}'.format(batch_ratio))
        try:
            if plot:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1

            for epoch in range(epochs):
                epoch_time = time.time()
                print("\n--> Epoch: {}/{}".format(
                    epoch,
                    epochs - 1
                ))

                for i in range(int(1 / batch_ratio)):
                    size = train_x.shape[0]
                    b_x = train_x[int(size * (i * batch_ratio)):
                                  int(size * ((i + 1) * batch_ratio))]
                    b_y = train_y[int(size * (i * batch_ratio)):
                                  int(size * ((i + 1) * batch_ratio))]
                    summary, opt = sess.run([self.summaries, self.opt],
                                   feed_dict=self.fill_feed_dict(
                                                                  b_x,
                                                                  b_y,
                                                                  self.learning_rate))
                    epoch_loss, epoch_acc = sess.run([self.cost, self.accuracy],
                                         feed_dict=self.fill_feed_dict(
                                                                      b_x,
                                                                      b_y,
                                                                      self.learning_rate))
                    sys.stdout.flush()
                    sys.stdout.write('\r\tDone {:.1f}% of the epoch'.format
                                     (100 * (i + 1) * batch_ratio))

                print("\nLoss= " + "{:.6f}".format(epoch_loss) + ", Accuracy= " + \
                      "{:.5f}".format(epoch_acc))
                summary_writer.add_summary(summary, epoch)
                train_accuracy, train_error = sess.run(
                                                [self.accuracy, self.cost],
                                                feed_dict=self.fill_feed_dict(
                                                                        train_x,
                                                                        train_y,
                                                                        self.learning_rate))
                self.record['epoch'].append(epoch)
                self.record['train_error'].append(train_error)
                self.record['train_accuracy'].append(train_accuracy)
                train_epoch = time.time() - epoch_time
                print("\n\ttrain error: {:.6f} |"" train accuracy: {:.6f} in {:.2f}s".format(
                    float(train_error),
                    float(train_accuracy),
                    train_epoch))
                if val_x is not None:
                    validation_accuracy, validation_error = sess.run(
                                                    [self.accuracy, self.cost],
                                                    feed_dict=self.fill_feed_dict(
                                                                            val_x,
                                                                            val_y,
                                                                            self.learning_rate))
                    self.record['validation_error'].append(validation_error)
                    self.record['validation_accuracy'].append(validation_accuracy)
                    valid_epoch = time.time() - epoch_time
                    print("\tvalid error: {:.6f} | valid accuracy: {:.6f} in {:.2f}s".format(
                        float(validation_error),
                        float(validation_accuracy),
                        valid_epoch))

        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")
        finally:
            self.timestamp = get_timestamp()
    def save_model(self, sess, saver, save_path='models'):
        """
        Will save the model

        Args:
            save_path: the location where you want to save the model and params
        """
        if not os.path.exists(save_path):
            print('Path not found, creating {}'.format(save_path))
            os.makedirs(save_path)

        file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                          self.name))
        if self.input_network is not None:
            if not hasattr(self.input_network['network'], 'save_name'):
                self.input_network['network'].save_model(sess, saver, file_path)

        self.save_name = '{}\{}_model'.format(file_path, self.name)
        print('Saving model as: {}'.format(self.save_name))

        saver.save(sess, self.save_name)

        self.save_metadata(file_path)

    @classmethod
    def load_model(self, sess, load_path, model_name):
        """
        Restores the model to the tensorflow sessionself

        Args:
            load_path: the location/directory where the model files are saved
                        (Eg: 'models/2018_06_13_173841_1_dense/')
            model_name: the name of the meta/model file to be loaded
                        (Eg: '1_dense_model')
                        retrieves: models/2018_06_13_173841_1_dense/1_dense_model.meta

        """
        'models/2018_06_13_173841_1_dense/1_dense_model.meta'
        print('Loading model from: {}'.format(load_path))
        #import the meta file (contains the model)
        model_instance = tf.train.import_meta_graph('{}{}.meta'.format(load_path, model_name))
        model_instance.restore(sess, tf.train.latest_checkpoint(load_path))
        print("Model restored from file: %s" % load_path)
        return model_instance

    def save_record(self, save_path='records'):
        """
        Will save the training records to file to be loaded up later.

        Args:
            save_path: the location where you want to save the records
        """
        if self.record is not None:
            if not os.path.exists(save_path):
                print('Path not found, creating {}'.format(save_path))
                os.makedirs(save_path)

            file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                              self.name))
            print('Saving records as: {}_stats.pickle'.format(file_path))
            with open('{}_stats.pickle'.format(file_path), 'w') as output:
                pickle.dump(self.record, output, -1)
        else:
            print("No record to save. Train the model first.")

    def save_metadata(self, files_path='models'):
        """
        Will save network configuration alongside weights.

        Args:
            file_path: the npz file path without the npz
        """
        config = {
            "{}".format(files_path): {
                "input_dim": self.input_dim,
                "input_var": "{}".format(self.input_var),
                "y": "{}".format(self.y),
                "config": self.config,
                "num_classes": self.num_classes,
                "input_network": {
                    'network': None,
                    'layer': None
                }
            }
        }
        if self.input_network:
            config["{}".format(files_path)]["input_network"]['network'] = \
                self.input_network['network'].save_name
            config["{}".format(files_path)]["input_network"]['layer'] = \
                self.input_network['layer']

        json_file = "{}\metadata.json".format(files_path)
        print('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as f:
            json.dump(config, f)



if __name__ == "__main__":
    pass
