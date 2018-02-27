from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import numpy as np
import argparse

# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='Breakout-v0', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.0001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=10000, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Iteration to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=250000, help='Number of frame before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--eps_decay', default=4000000,
                    help='Number of frames needed to decay epsilon to the lowest value', dest='eps_decay', type=int)
parser.add_argument('--lr_decay', default=80000000,
                    help='Number of frames needed to decay lr to the lowest value', dest='lr_decay', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps in Q-learning', dest='n_step', type=int)
parser.add_argument('--th_comp_fix', default=True,
                    help='Sets different Theano compiledir for each process', dest='th_fix', type=bool)
# -----
args = parser.parse_args()


import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

# -----


def build_network(input_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense

    x = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(x)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    v = Dense(output_shape, activation='linear')(h)
    return Model(inputs=x, outputs=v)


# -----

class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, screen=(84, 84), swap_freq=200):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.batch_size = batch_size

        self.action_value = build_network(self.observation_shape, action_space.n)
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')

        self.losses = deque(maxlen=25)
        self.q_values = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.frames = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        self.action_value.optimizer.lr.set_value(learning_rate)
        frames = len(last_observations)
        self.frames += frames
        # -----
        targets = self.action_value.predict_on_batch(last_observations)
        # -----
        targets[self.unroll, actions] = rewards
        # -----
        loss = self.action_value.train_on_batch(last_observations, targets)
        self.losses.append(loss)
        self.q_values.append(np.mean(targets))
        print('\rIter: %8d; Lr: %8.7f; Loss: %7.4f; Min: %7.4f; Max: %7.4f; Avg: %7.4f --- Q-value; Min: %7.4f; Max: %7.4f; Avg: %7.4f' % (
            self.frames, learning_rate, loss, min(self.losses), max(self.losses), np.mean(self.losses),
            np.min(self.q_values), np.max(self.q_values), np.mean(self.q_values)), end='')
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


def learn_proc(global_frame, mem_queue, weight_dict):
    import os
    pid = os.getpid()
    if args.th_fix:
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0,' + \
                                     'compiledir=th_comp_learn'
    # -----
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    lr_decay = args.lr_decay
    # -----
    env = gym.make(args.game)
    agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq)
    # -----
    if checkpoint > 0:
        print(' %5d> Loading weights from file' % (pid,))
        agent.action_value.load_weights('model-%d.h5' % (checkpoint,))
        # -----
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.action_value.get_weights()
    print(' %5d> Setting weights in dict' % (pid,))
    # -----
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    # -----
    idx = 0
    agent.frames = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        # -----
        last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.000000001, learning_rate * (1. - agent.frames / lr_decay))
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            global_frame.value = agent.frames
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.action_value.get_weights()
                weight_dict['update'] += 1
        # -----
        save_counter -= 1
        if save_counter % save_freq == 0:
            agent.action_value.save_weights('model-%d.h5' % (agent.frames,), overwrite=True)


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), n_step=8, discount=0.99):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.action_value = build_network(self.observation_shape, action_space.n)
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)
        # -----
        self.n_step_observations = deque(maxlen=n_step)
        self.n_step_actions = deque(maxlen=n_step)
        self.n_step_rewards = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # -----
        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = np.max(self.action_value.predict(self.observations[None, ...]))
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    def choose_action(self, epsilon=0.0):
        if np.random.random() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.action_value.predict(self.observations[None, ...]))

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


def generate_experience_proc(global_frame, mem_queue, weight_dict, no, epsilon):
    import os
    pid = os.getpid()
    if args.th_fix:
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                     'compiledir=th_comp_act_' + str(no)
    # -----
    batch_size = args.batch_size
    # -----
    print(' %5d> Process started with %6.3f' % (pid, epsilon))
    # -----
    env = gym.make(args.game)
    agent = ActingAgent(env.action_space, n_step=args.n_step)

    if args.checkpoint > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.action_value.load_weights('model-%d.h5' % (args.checkpoint,))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.action_value.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    best_score, last_update, frames = 0, 0, 0
    avg_score = deque(maxlen=20)
    stop_decay = global_frame.value > args.eps_decay

    while True:
        done = False
        episode_reward = 0
        last_op, op_count = 0, 0
        observation = env.reset()
        agent.init_episode(observation)

        # -----
        while not done:
            frames += 1
            if not stop_decay:
                frame_tmp = global_frame.value
                decayed_epsilon = max(epsilon, epsilon + (1. - epsilon) * (
                                        args.eps_decay - frame_tmp) / args.eps_decay)
                stop_decay = frame_tmp > args.eps_decay
            # -----
            action = agent.choose_action(decayed_epsilon)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            best_score = max(best_score, episode_reward)
            # -----
            agent.sars_data(action, reward, observation, done, mem_queue)
            # -----
            if action == last_op:
                op_count += 1
            else:
                op_count, last_op = 0, action
            # -----
            if op_count > 100:
                agent.reset()  # reset agent memory
                break
            # -----
            if frames % 2000 == 0:
                print(' %5d> Epsilon: %9.6f; Best score: %4d; Avg: %9.3f' % (
                    pid, decayed_epsilon, best_score, np.mean(avg_score)))
            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.action_value.set_weights(weight_dict['weights'])
        # -----
        avg_score.append(episode_reward)


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    manager = Manager()
    weight_dict = manager.dict()
    global_frame = manager.Value('i', args.checkpoint)
    mem_queue = manager.Queue(args.queue_size)

    eps = [0.1, 0.01, 0.5]
    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            p = pool.apply_async(generate_experience_proc,
                             args=(global_frame, mem_queue, weight_dict, i, eps[i % len(eps)]))
            # Get an eventual exception
            p.get()

        p = pool.apply_async(learn_proc, args=(global_frame, mem_queue, weight_dict))
        # Get an eventual exception
        p.get()

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
