import tensorflow as tf
import numpy as np
import os


class BaseAgent(object):
    def __init__(self, hp, logdir, log=True):
        self.n_actions = hp.n_actions
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


class DQNagent(BaseAgent):
    def __init__(self, hp, logdir, log=True):
        self.model = hp.model
        self.obs_shape = hp.obs_shape
        super(DQNagent, self).__init__(hp, logdir, log)

    def _placeholders(self):
        self.observation = tf.placeholder(tf.float32, shape=self.obs_shape)
        self.action = tf.placeholder(tf.int32, shape=(None, ))
        self.target_q = tf.placeholder(tf.float32, shape=(None, ))

    def _model(self):
        if self.model == "cnn":
            conv_1 = tf.layers.conv2d(self.observation, filters=32, kernel_size=8, strides=4, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_1 = tf.nn.relu(conv_1)
            conv_2 = tf.layers.conv2d(relu_1, filters=64, kernel_size=4, strides=2, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_2 = tf.nn.relu(conv_2)
            conv_3 = tf.layers.conv2d(relu_2, filters=64, kernel_size=3, strides=1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_3 = tf.nn.relu(conv_3)
            flatten_1 = tf.layers.flatten(relu_3)
            dense_1 = tf.layers.dense(flatten_1, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_4 = tf.nn.relu(dense_1)
            self.q = tf.layers.dense(relu_4, self.n_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())

        elif self.model == "mlp":
            dense_1 = tf.layers.dense(self.observation, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_1 = tf.nn.relu(dense_1)
            dense_2 = tf.layers.dense(relu_1, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_2 = tf.nn.relu(dense_2)
            self.q = tf.layers.dense(relu_2, self.n_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.q_max = tf.reduce_max(self.q, axis=1)
        self.best_action = tf.argmax(self.q, axis=1)

        mask = tf.one_hot(self.action, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
        q_sa = tf.boolean_mask(self.q, mask)
        self.loss = tf.losses.mean_squared_error(self.target_q, q_sa)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def choose_action(self, observation, epsilon=0):
        if np.random.random() <= epsilon:
            return np.random.randint(self.n_actions, size=(1,))
        else:
            action = self.sess.run(self.best_action, feed_dict={ self.observation: observation })
            return action

    def get_q_max(self, observation):
        return self.sess.run(self.q_max, feed_dict={ self.observation: observation })

    def update_policy(self, observation, action, target_q):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={ self.observation: observation, self.action: action, self.target_q: target_q})
        return loss

    def _summaries(self):
        self.writer = tf.summary.FileWriter(self.logdir, self.g)

        self.epsilon = tf.placeholder(tf.float32)
        self.reward = tf.placeholder(tf.float32)
        self.mean_reward = tf.placeholder(tf.float32)
        self.mean_q = tf.reduce_mean(self.q)

        self.epsilon_summary = tf.summary.scalar('epsilon', self.epsilon)
        self.reward_summary = tf.summary.scalar('reward', self.reward)
        self.mean_reward_summary = tf.summary.scalar('mean reward [100 episodes]', self.mean_reward)
        self.mean_q_summary = tf.summary.scalar('sampled mean q value', self.mean_q)

    def log_epsilon(self, epsilon, t):
        eps_sum = self.sess.run(self.epsilon_summary, feed_dict={ self.epsilon: epsilon })
        self.writer.add_summary(eps_sum, t)

    def log_reward(self, reward, mean_reward, t):
        rew_sum, mean_sum = self.sess.run([self.reward_summary, self.mean_reward_summary], feed_dict={ self.reward: reward, self.mean_reward: mean_reward })
        self.writer.add_summary(rew_sum, t)
        self.writer.add_summary(mean_sum, t)

    def log_q(self, observation, t):
        q_sum = self.sess.run(self.mean_q_summary, feed_dict={ self.observation: observation })
        self.writer.add_summary(q_sum, t)

