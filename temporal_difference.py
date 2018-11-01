import gym
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    desc = "Implementation of temporal difference methods for discrete OpenAI Gym enviornments"  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment. Must be discrete in action space and observation space. Observation space can be tuple of discrete spaces.', default='FrozenLake-v0')

    parser.add_argument('--method', type=str, default='q_learning', choices=['sarsa', 'q_learning'], help='Temporal difference method to determine action-value function and policy.')

    parser.add_argument('--episodes', type=int, default=10000, help='Total number of game episodes')

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    parser.add_argument('--gamma', type=float, default=0.95, help='Discout factor.')

    parser.add_argument('--epsilon', type=float, default=1.0, help='Starting value for epsilon, the random action probability.')

    parser.add_argument('--epsilon_taper', type=float, default=0.01, help='Rate to decrease epsilon over each episode.')

    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate.')

    parser.add_argument('--alpha_taper', type=float, default=0.01, help='Rate to decrease alpha over each visit to a state-action pair.')

    args = parser.parse_args()

    return args

class TDagent(object):
    def __init__(self, env, args=None):
        self.states = self._init_states(env)
        self.nA = env.action_space.n
        self.method = args.method
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = { s: np.random.randint(self.nA) for s in self.states }
        self.behavior = {}
        self.count = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_taper = args.epsilon_taper
        self.alpha = args.alpha
        self.alpha_taper = args.alpha_taper
        self.t = 0
        self.set_behavior()

    def _init_states(self, env):
        if type(env.observation_space) == gym.spaces.Tuple:
            states = list(product(*[list(range(space.n)) for space in env.observation_space.spaces]))
            pass
        else:
            states = list(range(env.observation_space.n))
        return states

    def set_behavior(self):
        self.t += 1
        epsilon = self.epsilon/(1 + self.epsilon_taper * self.t)
        for s in self.states:
            self.behavior[s] = np.ones(self.nA, dtype=float) * self.epsilon/self.nA
            self.behavior[s][self.policy[s]] += (1-self.epsilon)

    def choose_action(self, s):
        return np.random.choice(self.nA, p=self.behavior[s])

    def update_policy(self, state, action, reward, next_state, next_action=None):
        self._update_Q(state, action, reward, next_state, next_action)
        self._set_policy()

    def _update_Q(self, state, action, reward, next_state, next_action):
        next_Q = { 
            "sarsa": self.Q[next_state][next_action], 
            "q_learning": np.argmax(self.Q[next_state])
        }[self.method]
        self.count[state][action] += 1
        alpha = self.alpha/(1 + self.alpha_taper * self.count[state][action])
        self.Q[state][action] += alpha * (reward + self.gamma*next_Q - self.Q[state][action])

    def _set_policy(self):
        for s in self.states:
            self.policy[s] = np.argmax(self.Q[s])


def main():
    args = parse_args()

    print('Creating game environment for {}...'.format(args.env))
    env = gym.make(args.env)

    print('Creating agent...')
    agent = TDagent(env=env, args=args)

    print('Training agent...')
    episode_reward = []
    for e in range(args.episodes):
        total_reward = 0
        done = False
        state = env.reset()
        action = agent.choose_action(state)
        while not done:
            if args.render: 
                env.render()
            next_state, reward, done, _ = env.step(action)
            next_action = agent.choose_action(state)
            agent.update_policy(state, action, reward, next_state, next_action)
            total_reward += reward
            state = next_state
            action = next_action
        agent.set_behavior()
        episode_reward.append(total_reward)
    print('Average Reward from {} Episodes: {}'.format(args.episodes, np.mean(episode_reward)))
    plt.plot(episode_reward, 'o')
    plt.show()


if __name__ == '__main__':
    main()

