import gym
import tensorflow as tf
import numpy as np
import argparse

from libs.agents import BaseAgent
from libs.wrappers import wrap_env
from libs.utils import ExperienceBuffer, Transition, make_logdir


def parse_args():
    desc = "Implementation of Deep Q learning for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment.', default="PongNoFrameskip-v4")

    parser.add_argument('--buffer_size', type=int, help='Maximum capacity of experience replay buffer', default=10000)

    parser.add_argument('--sample_size', type=int, help='Number of transitions to sample when updating network', default=32)

    parser.add_argument('--sync_freq', type=int, help='How often to sync target and model networks', default=10000)

    parser.add_argument('--epsilon_start', type=float, help='Starting probability of taking a random action', default=1.0)

    parser.add_argument('--epsilon_final', type=float, help='Starting probability of taking a random action', default=0.02)

    parser.add_argument('--epsilon_decay', type=int, help='Number of frames to decay epsilon', default=10**6)

    parser.add_argument('--lr', type=float, help='Learning Rate.', default=1e-4)

    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)

    parser.add_argument('--total_frames', type=float, help='Total number of frames to simulate', default=1000000)

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    args = parser.parse_args()

    return args


class HP(object):
    def __init__(self, env, args):
        self.input_shape = (None, *env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.lr = args.lr


class DQNagent(BaseAgent):
    def _placeholders(self):
        self.observation = tf.placeholder(tf.float32, shape=self.input_shape)
        self.action = tf.placeholder(tf.int32, shape=(None, ))
        self.q_target = tf.placeholder(tf.float32, shape=(None, ))

    def _model(self):
        if len(self.input_shape) == 4:
            latent = self._cnn()
            dense_1 = tf.layers.dense(latent, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu_4 = tf.nn.relu(dense_1)
            self.q = tf.layers.dense(relu_4, self.num_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            latent = self._mlp()
            self.q = tf.layers.dense(input_stack, self.num_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.q_max = tf.reduce_max(self.q, axis=1)
        self.best_action = tf.argmax(self.q, axis=1)

        mask = tf.one_hot(self.action, self.num_actions, on_value=True, off_value=False, dtype=tf.bool)
        q_sa = tf.boolean_mask(self.q, mask)
        self.loss = tf.losses.mean_squared_error(self.q_target, q_sa)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def choose_action(self, observation, epsilon=0):
        if np.random.random() <= epsilon:
            return np.random.randint(self.num_actions, size=(1,))
        else:
            action = self.sess.run(self.best_action, feed_dict={ self.observation: observation })
            return action

    def get_q_max(self, observation):
        return self.sess.run(self.q_max, feed_dict={ self.observation: observation })

    def update_policy(self, observation, action, q_target, t):
        q_sum, loss, _ = self.sess.run([self.mean_q_summary, self.loss, self.train_op], feed_dict={ self.observation: observation, self.action: action, self.q_target: q_target})
        self.writer.add_summary(q_sum, t)

    def _summaries(self):
        self.writer = tf.summary.FileWriter(self.logdir, self.g)

        self.epsilon = tf.placeholder(tf.float32)
        self.reward = tf.placeholder(tf.float32)
        self.mean_reward = tf.placeholder(tf.float32)
        self.mean_q = tf.reduce_mean(self.q)

        self.epsilon_summary = tf.summary.scalar('epsilon', self.epsilon)
        self.reward_summary = tf.summary.scalar('reward', self.reward)
        self.mean_reward_summary = tf.summary.scalar('mean reward [100 episodes]', self.mean_reward)
        self.mean_q_summary = tf.summary.scalar('mean q value', self.mean_q)

    def log_epsilon(self, epsilon, t):
        eps_sum = self.sess.run(self.epsilon_summary, feed_dict={ self.epsilon: epsilon })
        self.writer.add_summary(eps_sum, t)

    def log_reward(self, reward, mean_reward, t):
        rew_sum, mean_sum = self.sess.run([self.reward_summary, self.mean_reward_summary], feed_dict={ self.reward: reward, self.mean_reward: mean_reward })
        self.writer.add_summary(rew_sum, t)
        self.writer.add_summary(mean_sum, t)


def calc_targets(transitions, target_agent, gamma):
    transitions = [(t.state, t.action, t.reward, t.done, t.next_state) for t in transitions]
    states, actions, rewards, dones, next_states = zip(*transitions)
    q_targets = target_agent.get_q_max(next_states)
    q_targets = rewards + gamma * q_targets * np.invert(dones)
    return states, actions, q_targets


def main():
    logdir = make_logdir('dqn')

    args = parse_args()
    env = gym.make(args.env)
    if env.observation_space.shape == (210, 160, 3):
        env = wrap_env(env)
    hp = HP(env, args)
    exp_buffer = ExperienceBuffer(args.buffer_size)

    agent = DQNagent(hp, logdir)
    checkpoint = agent.save_model()
    target_agent = DQNagent(hp, logdir, log=False)
    target_agent.load_model(checkpoint)

    t = 0
    rewards = []
    mean_reward_100 = 0
    episode = 0
    while t < args.total_frames:
        episode += 1
        o = env.reset()
        done = False
        total_reward = 0
        while not done:
            if args.render: env.render()
            t += 1

            epsilon = max(args.epsilon_final, args.epsilon_start - t / args.epsilon_decay)
            agent.log_epsilon(epsilon, t)

            a = agent.choose_action([o], epsilon)
            next_o, r, done, _ = env.step(a[0])
            exp_buffer.append(Transition(o, a, r, done, next_o, total_return=None))
            total_reward += r
            o = next_o

            if t % args.sync_freq == 0:
                checkpoint = agent.save_model()
                target_agent.load_model(checkpoint)

            if len(exp_buffer) == args.buffer_size:
                transitions = exp_buffer.sample(args.sample_size)
                observations, actions, q_targets = calc_targets(transitions, target_agent, args.gamma)
                agent.update_policy(observations, np.array(actions).flatten(), q_targets, t)

        print('Episode: {}, Reward: {}'.format(episode, total_reward))
        rewards.append(total_reward)
        mean_reward_100 = np.mean(rewards[-100:])
        agent.log_reward(total_reward, mean_reward_100, t)

    agent.save_model()


if __name__ == '__main__':
    main()
