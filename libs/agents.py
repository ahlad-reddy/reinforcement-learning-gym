import tensorflow as tf
import numpy as np
import os


class BaseAgent(object):
    def __init__(self, hp, logdir, log=True):
        self.input_shape = hp.input_shape
        self.num_actions = hp.num_actions
        self.lr = hp.lr
        self.logdir = logdir
        self.log = log
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self._placeholders()
            self._model()
            if self.log: self._summaries()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=3)

    def _placeholders(self):
        pass

    def _model(self):
        pass

    def _summaries(self):
        pass

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def choose_action(self, observation):
        pass

    def save_model(self):
        save_path = self.saver.save(self.sess, os.path.join(self.logdir, 'model.ckpt'), global_step=self.global_step)
        print("Model saved in path: %s" % save_path)
        return save_path

    def load_model(self, model_path):
        print('Loading model from %s' % model_path)
        self.saver.restore(self.sess, model_path)

    def _cnn(self):
        conv_1 = tf.layers.conv2d(self.observation, filters=32, kernel_size=8, strides=4, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        relu_1 = tf.nn.relu(conv_1)
        conv_2 = tf.layers.conv2d(relu_1, filters=64, kernel_size=4, strides=2, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        relu_2 = tf.nn.relu(conv_2)
        conv_3 = tf.layers.conv2d(relu_2, filters=64, kernel_size=3, strides=1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        relu_3 = tf.nn.relu(conv_3)
        flatten_1 = tf.layers.flatten(relu_3)
        return flatten_1

    def _mlp(self):
        dense_1 = tf.layers.dense(self.observation, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())
        tanh_1 = tf.nn.tanh(dense_1)
        dense_2 = tf.layers.dense(tanh_1, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())
        tanh_2 = tf.nn.tanh(dense_2)
        return tanh_2





