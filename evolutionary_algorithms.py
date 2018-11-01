import gym
import numpy as np
import argparse
from copy import copy
import matplotlib.pyplot as plt

def parse_args():
    desc = "Implementation of evolutionary algorithms for continuous OpenAI Gym enviornments"  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment.', default='BipedalWalker-v2')

    parser.add_argument('--method', type=str, default='augmented_random_search', choices=['augmented_random_search'], help='')

    parser.add_argument('--episodes', type=int, default=1000, help='Total number of game episodes')

    parser.add_argument('--num_deltas', type=int, default=16, help='Number of agents to sample for each update')

    parser.add_argument('--num_best_deltas', type=int, default=16, help='Number of agents to sample for each update')

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    parser.add_argument('--noise', type=float, default=0.03, help='Noise scaling for weights')

    parser.add_argument('--alpha', type=float, default=0.02, help='Learning rate.')

    parser.add_argument('--seed', type=int, default=2018, help='Seed number')

    args = parser.parse_args()

    return args


class ARSagent(object):
    def __init__(self, shape, noise):
        self.weights = np.zeros(shape)
        self.noise = noise

    def choose_action(self, state, delta):
        action = (self.weights + self.noise * delta).dot(state)
        return action

class Normalizer():
    # Normalizes the inputs
    def __init__(self, nS):
        self.n = 0
        self.mean = np.zeros(nS)
        self.mean_diff = np.zeros(nS)
        self.var = np.zeros(nS)

    def normalize(self, state):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (state - self.mean) / self.n
        self.mean_diff += (state - last_mean) * (state - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        return (state - self.mean) / np.sqrt(self.var)


class Experiment(object):
    def __init__(self, args, agent_cn):
        print('Creating game environment for {}...'.format(args.env))
        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.shape[0]
        self.shape = (self.nA, self.nS)
        self.episodes = args.episodes
        self.agent = agent_cn(self.shape, args.noise)
        self.num_deltas = args.num_deltas
        self.num_best_deltas = args.num_best_deltas
        self.normalizer = Normalizer(self.nS)
        self.alpha = args.alpha
        self.render = args.render
        self.seed = 1

    def run(self):
        best_agent_rewards = []
        for e in range(self.episodes):
            results = self._test_deltas()
            self._update_policy(results)
            reward = self._run_episode(delta=np.zeros(self.shape), render=self.render)
            print('Step: {}, Reward: {}'.format(e, reward))
            best_agent_rewards.append(reward)
        plt.plot(best_agent_rewards, 'ro')
        plt.show()

    def _test_deltas(self):
        deltas = [np.random.randn(*self.shape) for _ in range(self.num_deltas)]
        results = []
        for delta in deltas:
            r1 = self._run_episode(delta)
            r2 = self._run_episode(-delta)
            results.append((r1, r2, delta))
        return results

    def _run_episode(self, delta, render=False):
        done = False
        total_reward = 0
        state = self.env.reset()
        while not done:
            if render: self.env.render()
            state = self.normalizer.normalize(state)
            state, reward, done, _ = self.env.step(self.agent.choose_action(state, delta))
            total_reward += max(min(reward, 1), -1)
        return total_reward

    def _update_policy(self, results):
        grad = np.zeros(self.shape)
        results = sorted(results, key=lambda result: max(result[0], result[1]), reverse=True)[:self.num_best_deltas]
        for r1, r2, delta in results:
            grad += (r1 - r2) * delta
        sigma_rewards = np.array([[r1, r2] for r1, r2, delta in results]).std()
        self.agent.weights += (self.alpha / (sigma_rewards * self.num_best_deltas)) * grad



def main():
    args = parse_args()
    np.random.seed(args.seed)

    experiment = Experiment(args, ARSagent)
    experiment.run()


if __name__ == '__main__':
    main()

