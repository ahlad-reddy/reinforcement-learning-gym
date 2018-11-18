import gym
import tensorflow as tf
import numpy as np
import argparse
from collections import deque, namedtuple
from libs.agents import BaseAgent
from libs.utils import make_logdir, ExperienceBuffer


def parse_args():
    desc = "Implementation of policy gradient algorithms for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment.', default="CartPole-v1")

    parser.add_argument('--episodes', type=int, help='Total number of episodes to run', default=500)

    parser.add_argument('--buffer_size', type=int, help='Maximum capacity of experience replay buffer', default=10000)

    parser.add_argument('--lr', type=float, help='Learning Rate.', default=1e-4)

    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    args = parser.parse_args()

    return args


class HP(object):
    def __init__(self, env, args):
        self.input_shape = (None, *env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.lr = args.lr


class Transition(object):
    def __init__(self, state, action, reward, done, next_state, total_return=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state
        self.total_return = total_return


class REINFORCE(BaseAgent):
    def _placeholders(self):
        self.observation = tf.placeholder(tf.float32, shape=self.input_shape)
        self.action = tf.placeholder(tf.int32, shape=(None, ))
        self.advantage = tf.placeholder(tf.float32, shape=(None, ))

    def _model(self):
        latent = self._mlp()
        self.logits = tf.layers.dense(latent, self.num_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.p = tf.nn.softmax(self.logits)

        action_mask = tf.one_hot(self.action, self.num_actions, on_value=True, off_value=False, dtype=tf.bool)
        action_p = tf.boolean_mask(self.p, action_mask)
        self.loss = -tf.reduce_sum(self.advantage * tf.log(action_p))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def sample_action(self, observation):
        p = self.sess.run(self.p, feed_dict={ self.observation: observation })
        return np.random.choice(self.num_actions, p=p.flatten())

    def update_policy(self, observation, action, advantage):
        loss, loss_sum, _, gs = self.sess.run([self.loss, self.loss_summary, self.train_op, self.global_step], feed_dict={ self.observation: observation, self.action: action, self.advantage: advantage })
        self.writer.add_summary(loss_sum, gs)
        return gs

    def _summaries(self):
        self.writer = tf.summary.FileWriter(self.logdir, self.g)

        self.reward = tf.placeholder(tf.float32)
        self.mean_reward = tf.placeholder(tf.float32)
        self.baseline = tf.placeholder(tf.float32)

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.reward_summary = tf.summary.scalar('reward', self.reward)
        self.mean_reward_summary = tf.summary.scalar('mean reward [100 episodes]', self.mean_reward)
        self.baseline_summary = tf.summary.scalar('baseline', self.baseline)

    def log_reward(self, reward, mean_reward, baseline, gs):
        rew_sum, mean_sum, base_sum = self.sess.run([self.reward_summary, self.mean_reward_summary, self.baseline_summary], feed_dict={ self.reward: reward, self.mean_reward: mean_reward, self.baseline: baseline })
        self.writer.add_summary(rew_sum, gs)
        self.writer.add_summary(mean_sum, gs)
        self.writer.add_summary(base_sum, gs)


def main():
    logdir = make_logdir('pg')

    args = parse_args()
    env = gym.make(args.env)
    hp = HP(env, args)
    exp_buffer = ExperienceBuffer(args.buffer_size)
    agent = REINFORCE(hp, logdir)

    rewards = []
    for episode in range(args.episodes):
        trajectory = []
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            if args.render: 
                env.render()
            action = agent.sample_action([observation])
            next_obs, reward, done, _ = env.step(action)
            trajectory.append(Transition(observation, action, reward, done, next_obs, total_return=None))
            observation = next_obs
            total_reward += reward
        rewards.append(total_reward)

        for i, transition in enumerate(trajectory):
            transition.total_return = sum([(args.gamma**j) * t.reward for j, t in enumerate(trajectory[i:])])
            exp_buffer.append(transition)
        for t in trajectory:
            advantage = t.total_return - exp_buffer.mean_value()
            gs = agent.update_policy([t.state], [t.action], [advantage])

        print('Episode: {}/{}, Total Reward: {}'.format(episode, args.episodes, total_reward))
        agent.log_reward(total_reward, np.mean(rewards[-100:]), exp_buffer.mean_value(), gs)


if __name__ == '__main__':
    main()
