import gym
import neat
import numpy as np


env = gym.make('Acrobot-v1')

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(net.activate(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward
        genome.fitness = total_reward

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'GymNEAT')

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False))
winner = p.run(eval_genomes, 1000)

net = neat.nn.FeedForwardNetwork.create(winner, config)
state = env.reset()
total_reward = 0
done = False
while not done:
    env.render()
    action = np.argmax(net.activate(state))
    state, reward, done, _ = env.step(action)
    total_reward += reward
print('Total Reward:', total_reward)