from collections import deque, namedtuple
import numpy as np
import os
import glob
import time


Transition = namedtuple('Transition', field_names=['state', 'action', 'reward', 'done', 'next_state', 'advantage'])


class ExperienceBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append_transition(self, state, action, reward, done, next_state):
        t = Transition(state, action, reward, done, next_state, advantage=None)
        self.buffer.append(t)

    def sample(self, sample_size):
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        return [self.buffer[i] for i in indices]

    def mean_value(self):
        return np.mean([t.reward for t in self.buffer])


def make_logdir():
    if not os.path.exists('logdir'): os.mkdir('logdir')
    logdir = 'logdir/dqn-{}-{:03d}'.format(time.strftime("%y%m%d"), len(glob.glob('logdir/*')))
    os.mkdir(logdir)
    print('Saving to results to {}'.format(logdir))
    return logdir