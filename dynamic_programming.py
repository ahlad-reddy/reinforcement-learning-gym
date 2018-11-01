import gym
import argparse
import numpy as np
import time
from copy import deepcopy


def parse_args():
    desc = "Implementation of dynamic programming methods for OpenAI Gym enviornments with known transition probabilities"  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment. Must be discrete in action space and observation space. Tested with FrozenLake-v0, FrozenLake8x8-v0, & Taxi-v2.', default='FrozenLake-v0')

    parser.add_argument('--method', type=str, default='policy_iteration', choices=['value_iteration', 'policy_iteration'], help='Dynamic programming method to calculate value function and policy.')

    parser.add_argument('--episodes', type=int, default=1000, help='Total number of game episodes')

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    parser.add_argument('--gamma', type=float, default=0.9, help='Discout factor.')

    parser.add_argument('--theta', type=float, default=1e-3, help='Threshold when calculating value function')

    parser.add_argument('--max_iterations', type=int, default=10000, help='Maximum number of iterations when calculating value function.')

    args = parser.parse_args()

    return args


class DPagent(object):
    def __init__(self, env, args):
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.P = env.env.P
        self.V = { s: 0 for s in range(self.nS) }
        self.method = args.method
        self.policy = { s: np.random.randint(self.nA) for s in range(self.nS) }
        self.gamma = args.gamma
        self.theta = args.theta
        self.max_iterations = args.max_iterations
        self._create_policy()

    def _action_value(self, state):
        A = np.zeros(self.nA)
        for a in range(self.nA):
            A[a] = sum([p * (reward + self.gamma * self.V[next_state]) for (p, next_state, reward, d) in self.P[state][a]])
        return A

    def _value_iteration(self):
        for i in range(self.max_iterations):
            delta = 0
            for s in range(self.nS):
                if self.method == "value_iteration":
                    v = max(self._action_value(s))
                elif self.method == "policy_iteration":
                    v = self._action_value(s)[self.policy[s]]
                delta = max(delta, abs(self.V[s] - v))
                self.V[s] = v
            if delta < self.theta:
                break
        if i == self.max_iterations-1: print('Reached maximum iterations...')
        for s in range(self.nS):
            self.policy[s] = np.argmax(self._action_value(s))

    def _policy_iteration(self):
        for i in range(self.max_iterations):
            old_policy = deepcopy(self.policy)
            self._value_iteration()
            same_policy = all([old_policy[s] == self.policy[s] for s in range(self.nS)])
            if same_policy:
                break
        if i == self.max_iterations-1: print('Reached maximum iterations...')

    def _create_policy(self):
        start_time = time.time()
        if self.method == "value_iteration":
            self._value_iteration()
        elif self.method == "policy_iteration":
            self._policy_iteration()
        print('Policy calculation finished in {:.2f} seconds...'.format(time.time()-start_time))


def main():
    args = parse_args()

    print('Creating game environment for {}...'.format(args.env))
    env = gym.make(args.env)

    print('Creating agent...')
    agent = DPagent(env=env, args=args)

    total_reward = 0
    for e in range(args.episodes):
        s = env.reset()
        done = False
        while not done:
            if args.render: 
                env.render()
            s, reward, done, _ = env.step(agent.policy[s])
            total_reward += reward
    print('Total Reward from {} Episodes: {}'.format(args.episodes, total_reward))


if __name__ == '__main__':
    main()
