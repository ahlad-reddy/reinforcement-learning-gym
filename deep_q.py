import gym
import tensorflow as tf
import numpy as np
import argparse
from collections import deque
from DL.agents import DCQNagent, DQNagent
from libs.wrappers import make_env
import os
import time


def parse_args():
    desc = "Implementation of Deep Q learning for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment.', default="PongNoFrameskip-v4")

    parser.add_argument('--buffer_size', type=int, help='Maximum capacity of experience replay buffer', default=10000)

    parser.add_argument('--sample_size', type=int, help='Number of transitions to sample when updating network', default=32)

    parser.add_argument('--sync_freq', type=int, help='How often to sync target and model networks', default=1000)

    parser.add_argument('--sample_obs', type=int, help='Number of sample observations to log q value', default=32)

    parser.add_argument('--epsilon_start', type=float, help='Starting probability of taking a random action', default=1.0)

    parser.add_argument('--epsilon_final', type=float, help='Starting probability of taking a random action', default=0.02)

    parser.add_argument('--epsilon_decay', type=int, help='Number of frames to decay epsilon', default=10**5)

    parser.add_argument('--lr', type=float, help='Learning Rate.', default=1e-4)

    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)

    parser.add_argument('--mean_reward_goal', type=float, help='Discount factor', default=19.5)

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    args = parser.parse_args()

    return args


class HP(object):
    def __init__(self, env, args):
        self.n_actions = env.action_space.n
        self.lr = args.lr


class ExperienceBuffer(object):
    def __init__(self, args):
        self.buffer = deque(maxlen=args.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, sample_size):
        return np.random.choice(self.buffer, sample_size, replace=False)


def get_sample_observations(env, agent, num_samples):
    samples = []
    while len(samples) < num_samples:
        o = env.reset()
        done = False
        while not done:
            a = agent.choose_action([o])
            next_o, r, done, _ = env.step(a[0])
            samples.append(o)
            o = next_o
            if len(samples) == num_samples:
                break
    return samples


def calc_targets(transitions, target_agent, gamma):
    transitions = [(t['state'],t['action'],t['reward'],t['done'],t['next_state']) for t in transitions]
    states, actions, rewards, dones, next_states = zip(*transitions)
    target_q = target_agent.get_q_max(next_states)
    target_q = rewards + gamma * target_q * np.invert(dones)
    return states, actions, target_q


def main():
    if not os.path.exists('logdir'): os.mkdir('logdir')
    logdir = 'logdir/dqn-{}'.format(time.strftime("%y-%m-%dT%H-%M-%S"))
    os.mkdir(logdir)
    print('Saving to results to {}'.format(logdir))

    args = parse_args()
    # env = make_env(args.env)
    env = gym.make('LunarLander-v2')
    hp = HP(env, args)
    exp_buffer = ExperienceBuffer(args)

    agent = DQNagent(hp, logdir)
    checkpoint = agent.save_model()
    target_agent = DQNagent(hp, logdir, log=False)
    target_agent.load_model(checkpoint)

    sample_obs = get_sample_observations(env, agent, args.sample_obs)

    t = 0
    rewards = []
    mean_reward_100 = 0
    episode = 0
    while mean_reward_100 < args.mean_reward_goal:
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
            exp_buffer.append({ 'state': o, 'action': a, 'reward': r, 'done': done, 'next_state': next_o })
            total_reward += r
            o = next_o

            if len(exp_buffer) < args.buffer_size:
                continue

            if t % args.sync_freq == 0:
                checkpoint = agent.save_model()
                target_agent.load_model(checkpoint)

            transitions = exp_buffer.sample(args.sample_size)
            observations, actions, target_q = calc_targets(transitions, target_agent, args.gamma)
            loss = agent.update_policy(observations, np.array(actions).flatten(), target_q)
            # print('Frame: {}, Loss: {}'.format(t, loss))

        print('Episode: {}, Reward: {}'.format(episode, total_reward))
        rewards.append(total_reward)
        mean_reward_100 = np.mean(rewards[-100:])
        agent.log_reward(total_reward, mean_reward_100, t)
        agent.log_q(sample_obs, t)

    agent.save_model()


if __name__ == '__main__':
    main()
