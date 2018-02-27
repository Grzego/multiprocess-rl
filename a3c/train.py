from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import numpy as np
import h5py
import argparse

import traceback, functools, multiprocessing

def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            print('Exception in '+func.__name__)
            traceback.print_exc()
    return wrapped_func

# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='Breakout-v0', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=80000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=250000, help='Number of frames before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', dest='n_step', type=int)
parser.add_argument('--reward_scale', default=1., dest='reward_scale', type=float)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
parser.add_argument('--lstm', type=bool, default=False, help='Use a LSTM', dest='lstm')
# -----
args = parser.parse_args()


# -----


def build_network(input_shape, output_shape, lstm=False):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense, LSTM, TimeDistributed
    # -----
    if lstm:
        # Inspired from https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

        state = Input(shape=input_shape)
        h = TimeDistributed(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first'))(state)
        h = TimeDistributed(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first'))(h)
        h = TimeDistributed(Flatten())(h)
        h = TimeDistributed(Dense(256, activation='relu'))(h)
        h = LSTM(256)(h)

        value = Dense(1, activation='linear', name='value')(h)
        policy = Dense(output_shape, activation='softmax', name='policy')(h)

    else:
        state = Input(shape=input_shape)
        h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(state)
        h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)

        value = Dense(1, activation='linear', name='value')(h)
        policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    advantage = Input(shape=(1,))
    train_network = Model(inputs=[state, advantage], outputs=[value, policy])

    # from keras.utils import plot_model
    # plot_model(train_network, to_file='train_network_lstm.png', show_shapes=True)

    return value_network, policy_network, train_network, advantage


def policy_loss(advantage=0., beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss


# -----

class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, screen=(84, 84), swap_freq=200, lstm=False):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.time_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        if lstm:
            self.observation_shape = (self.time_depth,) + self.observation_shape
        self.batch_size = batch_size

        _, _, self.train_net, advantage = build_network(self.observation_shape, action_space.n, lstm=lstm)

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(advantage, args.beta)])

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        # -----
        values, policy = self.train_net.predict([last_observations, self.unroll])
        # -----
        self.targets.fill(0.)
        advantage = rewards - values.flatten()
        self.targets[self.unroll, actions] = 1.
        # -----
        loss = self.train_net.train_on_batch([last_observations, advantage], [rewards, self.targets])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val), end='')
        # -----
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


@trace_unhandled_exceptions
def learn_proc(envs, mem_queue, weight_dict, lstm=False):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu,compiledir=th_comp_learn'
    # -----
    print(' %5d> Learning process' % (pid,))
    # -----
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps
    # -----
    env = gym.make(args.game)
    agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq, lstm=lstm)
    # -----
    if checkpoint > 0:
        print(' %5d> Loading weights from file' % (pid,))
        agent.train_net.load_weights('model-%s-%d.h5' % (args.game, checkpoint,))
        # -----
    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    # -----
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    # -----
    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    try:
        while True:
            # -----
            last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
            idx = (idx + 1) % batch_size
            if idx == 0:
                lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
                updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
                if updated:
                    # print(' %5d> Updating weights in dict' % (pid,))
                    weight_dict['weights'] = agent.train_net.get_weights()
                    weight_dict['update'] += 1
            # -----
            save_counter -= 1
            if save_counter < 0:
                save_counter += save_freq
                agent.train_net.save_weights('model-%s-%d.h5' % (args.game, agent.counter,), overwrite=True)
    except Exception as e:
        print('Got exception ', e, ', closing')
        for env in envs:
            env.close()



class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), n_step=8, discount=0.99, lstm=False):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.time_depth = 1
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        if lstm:
            self.observation_shape = (self.time_depth,) + self.observation_shape

        self.value_net, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n, lstm=lstm)

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

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

    def init_episode(self, observation, lstm):
        for _ in range(self.past_range):
            self.save_observation(observation, lstm)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue, lstm):
        self.save_observation(observation, lstm)
        reward /= args.reward_scale
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
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    def choose_action(self):
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation, lstm):
        self.last_observations = self.observations[...]
        input_depth = self.time_depth if lstm else self.input_depth

        self.observations = np.roll(self.observations, -input_depth, axis=0)
        self.observations[-input_depth:, ...] = self.transform_screen(observation, lstm)

    def transform_screen(self, data, lstm):
        if lstm:
            return rgb2gray(imresize(data, self.screen))
        else:
            return rgb2gray(imresize(data, self.screen))[None, ...]


@trace_unhandled_exceptions
def generate_experience_proc(env, mem_queue, weight_dict, no, lstm=False):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu,compiledir=th_comp_act_' + str(no)
    # -----
    print(' %5d> Process started' % (pid,))
    # -----
    frames = 0
    batch_size = args.batch_size
    # -----
    agent = ActingAgent(env.action_space, n_step=args.n_step, lstm=lstm)

    if frames > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.load_net.load_weights('model-%s-%d.h5' % (args.game, frames))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    best_score = 0
    avg_score = deque([0], maxlen=25)

    last_update = 0
    try:
        while True:
            done = False
            episode_reward = 0
            op_last, op_count = 0, 0
            observation = env.reset()
            agent.init_episode(observation, lstm)

            # -----
            while not done:
                frames += 1
                action = agent.choose_action()
                observation, reward, done, _ = env.step(action)
                episode_reward += reward
                best_score = max(best_score, episode_reward)
                # -----
                agent.sars_data(action, reward, observation, done, mem_queue, lstm)
                # -----
                op_count = 0 if op_last != action else op_count + 1
                done = done or op_count >= 100
                op_last = action
                # -----
                if frames % 2000 == 0:
                    print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                        pid, best_score, np.mean(avg_score), np.max(avg_score)))
                if frames % batch_size == 0:
                    update = weight_dict.get('update', 0)
                    if update > last_update:
                        last_update = update
                        # print(' %5d> Getting weights from dict' % (pid,))
                        agent.load_net.set_weights(weight_dict['weights'])
            # -----
            avg_score.append(episode_reward)
    except Exception as e:
        print('Got exception ', e, ', closing')
        env.close()


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    if args.game.startswith('Super'):
        import super_mario

    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    pool = Pool(args.processes + 1, init_worker)
    envs = []

    try:
        for i in range(args.processes):
            env = gym.make(args.game)
            pool.apply_async(generate_experience_proc, (env, mem_queue, weight_dict, i, args.lstm))
            envs.append(env)

        pool.apply_async(learn_proc, (envs, mem_queue, weight_dict, args.lstm))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
