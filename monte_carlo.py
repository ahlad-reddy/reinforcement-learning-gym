import gym
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from libs import plotting


def parse_args():
    desc = "Implementation of monte carlo methods for discrete OpenAI Gym enviornments"  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment. Must be discrete in action space and observation space. Observation space can be tuple of descrete spaces.', default='FrozenLake-v0')

    parser.add_argument('--method', type=str, default='epsilon_greedy', choices=['epsilon_greedy', 'importance_sampling'], help='Monte Carlo method to determine action-value function and policy.')

    parser.add_argument('--visit_method', type=str, default='first_visit', choices=['first_visit', 'every_visit'], help='Method to handle multiple visits to the same state-action pair.')

    parser.add_argument('--episodes', type=int, default=10000, help='Total number of game episodes')

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    parser.add_argument('--gamma', type=float, default=0.9, help='Discout factor.')

    parser.add_argument('--epsilon', type=float, default=0.2, help='Random exploration probability.')

    args = parser.parse_args()

    if args.method == "importance_sampling": args.visit_method = "every_visit"

    return args


class MCagent(object):
    def __init__(self, env, args=None):
        self.states = self._init_states(env)
        self.nA = env.action_space.n
        self.method = args.method
        self.visit_method = args.visit_method
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = { s: np.random.randint(self.nA) for s in self.states }
        self.behavior = {}
        self.count = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self._set_behavior()

    def _init_states(self, env):
        if type(env.observation_space) == gym.spaces.Tuple:
            states = list(product(*[list(range(space.n)) for space in env.observation_space.spaces]))
            pass
        else:
            states = list(range(env.observation_space.n))
        return states

    def _set_behavior(self):
        for s in self.states:
            self.behavior[s] = np.ones(self.nA, dtype=float) * self.epsilon/self.nA
            self.behavior[s][self.policy[s]] += (1-self.epsilon)

    def choose_action(self, s):
        return np.random.choice(self.nA, p=self.behavior[s])

    def update_policy(self, trajectory):
        trajectory = self._calculate_advantages(trajectory)
        self._update_Q(trajectory)
        self._update_policy()
        self._set_behavior()

    def _calculate_advantages(self, trajectory):
        G = 0
        for i in reversed(range(len(trajectory))):
            G = trajectory[i][2] = trajectory[i][2] + self.gamma * G
        return trajectory

    def _update_Q(self, trajectory):
        W = 1
        visited_states = []
        if self.visit_method == "every_visit": 
            trajectory = trajectory[::-1]
        for s, a, G in trajectory:
            if self.visit_method == "every_visit" or (s, a) not in visited_states:
                visited_states.append((s, a))
                self.count[s][a] += W
                self.Q[s][a] += (W/self.count[s][a])*(G - self.Q[s][a])
                if self.method == "importance_sampling":
                    if a != self.policy[s]:
                        break
                    W *= 1/self.behavior[s][a]

    def _update_policy(self):
        for s in self.states:
            self.policy[s] = np.argmax(self.Q[s])


def main():
    args = parse_args()

    print('Creating game environment for {}...'.format(args.env))
    env = gym.make(args.env)

    print('Creating agent...')
    agent = MCagent(env=env, args=args)

    print('Training agent...')
    episode_rewards = []
    for e in range(args.episodes):
        s = env.reset()
        trajectory = []
        total_reward = 0
        done = False
        while not done:
            if args.render: 
                env.render()
            a = agent.choose_action(s)
            next_state, r, done, _ = env.step(a)
            trajectory.append([s, a, r])
            total_reward += r
            s = next_state
        agent.update_policy(trajectory)
        episode_rewards.append(total_reward)
    print('Average Reward from {} Episodes: {}'.format(args.episodes, np.mean(episode_rewards)))
    plt.plot(episode_rewards, 'ro')
    plt.show()

    if args.env == "Blackjack-v0":
        V = defaultdict(float)
        for state, actions in agent.Q.items():
            action_value = np.max(actions)
            V[state] = action_value
        plotting.plot_value_function(V, title="Optimal Value Function")


if __name__ == '__main__':
    main()

