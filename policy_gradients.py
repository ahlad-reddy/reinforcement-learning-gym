import gym
import tensorflow as tf
import numpy as np
import argparse
from collections import deque, namedtuple
from dqn.agents import BaseAgent


def parse_args():
    desc = "Implementation of policy gradient algorithms for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment.', default="CartPole-v0")

    parser.add_argument('--total_frames', type=int, help='Total number of frames to run', default=10000)

    parser.add_argument('--buffer_size', type=int, help='Maximum capacity of experience replay buffer', default=10000)

    parser.add_argument('--sample_size', type=int, help='Number of transitions to sample when updating network', default=32)

    parser.add_argument('--sample_obs', type=int, help='Number of sample observations to log q value', default=32)

    parser.add_argument('--lr', type=float, help='Learning Rate.', default=1e-4)

    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    args = parser.parse_args()

    return args


Transition = namedtuple('Transition', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer(object):
    def __init__(self, args):
        self.buffer = deque(maxlen=args.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, trajectory):
        for t in transitions:
            self.buffer.append(t)

    def sample(self, sample_size):
        return np.random.choice(self.buffer, sample_size, replace=False)

    def mean_value(self):
        return np.mean([t.reward for t in self.buffer])


class REINFORCE(BaseAgent):
    def _placeholders(self):
        self.observation = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.action = tf.placeholder(tf.float32, shape=(None, ))
        self.action_value = tf.placeholder(tf.float32, shape=(None, ))
        self.mean_value = tf.placeholder(tf.float32)

    def _model(self):
        dense_1 = tf.layers.dense(self.observation, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())
        relu_1 = tf.nn.relu(dense_1)
        self.logits = tf.layers.dense(relu_1, self.output_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.p = tf.layers.softmax(self.logits)

        action_mask = tf.one_hot(self.action, self.output_size, on_value=True, off_value=False, dtype=tf.bool)
        action_p = tf.boolean_mask(self.p, action_mask)
        self.loss = -tf.tensordot(tf.subtract(self.action_value, self.mean_value), tf.math.log(action_p))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def sample_action(self, observation):
        p = self.sess.run(self.p, feed_dict={ self.observation: observation })
        return np.random.choice(self.output_size, p=p)

    def update_policy(self, observation, action, action_value, mean_value):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={ self.observation: observation, self.action: action, self.action_value: action_value, self.mean_value: mean_value })
        return loss


def main():
    args = parse_args()
    env = gym.make(args.env)
    exp_buffer = ExperienceBuffer(args.buffer_size)
    agent = REINFORCE(args)

    for episode in range(args.total_episodes):
        trajectory = []
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            if args.render: 
                env.render()
            action = agent.sample_action(observation)
            next_obs, reward, done, _ = env.step(action)
            trajectoy.append(Transition(observation, action, reward, done, next_obs))
            observation = next_obs
            total_reward += reward

            if len(exp_buffer) < args.buffer_size:
                continue

            observations, actions, action_values, mean_value = exp_buffer.sample(args.sample_size)
            agent.update_policy(observations, actions, action_values, mean_value)

        print('Episode: {}/{}, Total Reward: {}'.format(episode, args.total_episodes, total_reward))
        # calculate discounted reward
        exp_buffer.append(trajectory)





if __name__ == '__main__':
    main()