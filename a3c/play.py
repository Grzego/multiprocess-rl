from keras.models import *
from keras.layers import *
from keras.optimizers import RMSprop
import gym
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np
from gym import error, version, logger
from gym.wrappers import Monitor
from gym.wrappers.monitoring import stats_recorder
from gym.wrappers.monitor import detect_training_manifests, monitor_closer, FILE_PREFIX
from scipy.misc import imresize
from skimage.color import rgb2gray
import six
import numpy as np
import argparse


class MarioStatsRecorder(stats_recorder.StatsRecorder):
    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super(MarioStatsRecorder, self).__init__(directory, file_prefix, autoreset, env_id)
        self.infos = []

    def after_step(self, observation, reward, done, info):
        self.info = info
        super(MarioStatsRecorder, self).after_step(observation, reward, done, info)

    def save_complete(self):
        if self.steps is not None:
            self.infos.append(self.info)
        super(MarioStatsRecorder, self).save_complete()

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
                'infos': self.infos,
            }, f, default=json_encode_np)


class MarioMonitor(Monitor):
    def _start(self, directory, video_callable=None, force=False, resume=False,
              write_upon_reset=False, uid=None, mode=None):
        """Copy from gym.Monitor"""
        if self.env.spec is None:
            logger.warn("Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable is False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(type(video_callable), video_callable))
        self.video_callable = video_callable

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.
 You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        self.stats_recorder = MarioStatsRecorder(directory, '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix), autoreset=self.env_semantics_autoreset, env_id=env_id)

        if not os.path.exists(directory): os.mkdir(directory)
        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)


def build_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(state)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    if add_lstm:
        h = LSTM(256)(h)

    value = Dense(1, activation='linear')(h)
    policy = Dense(output_shape, activation='softmax')(h)

    value_network = Model(input=state, output=value)
    policy_network = Model(input=state, output=policy)

    adventage = Input(shape=(1,))
    train_network = Model(inputs=state, outputs=[value, policy])

    return value_network, policy_network, train_network, adventage


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), add_lstm=False):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.replay_size = 32
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        _, self.policy, self.load_net, _ = build_network(self.observation_shape, action_space.n, add_lstm)

        self.load_net.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.

        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def choose_action(self, observation):
        self.save_observation(observation)
        policy = self.policy.predict(self.observations[None, ...])[0]
        policy /= np.sum(policy)  # numpy, why?
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='Breakout-v0', help='Name of openai gym environment', dest='game')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation', dest='evaldir')
parser.add_argument('--model', help='File with weights for model', dest='model')
parser.add_argument('--n-tests', type=int, default=20, help='Number of tests to run', dest='n_tests')
parser.add_argument('--lstm', type=bool, default=False, help='Use a LSTM', dest='add_lstm')


def main():
    args = parser.parse_args()
    if args.game.startswith('Super'):
        import super_mario
    # -----
    env = gym.make(args.game)
    if args.evaldir:
        if args.game.startswith('Super'):
            env = MarioMonitor(env, args.evaldir, video_callable=lambda episode_id: True)
        else:
            env = Monitor(env, args.evaldir, video_callable=lambda episode_id: True)
    # -----
    agent = ActingAgent(env.action_space, args.add_lstm)

    model_file = args.model

    agent.load_net.load_weights(model_file)

    game = 1
    episode_rewards = []
    for _ in range(args.n_tests):
        done = False
        episode_reward = 0
        noops = 0

        # init game
        observation = env.reset()
        agent.init_episode(observation)
        # play one game
        print('Game #%8d; ' % (game,), end='')
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # ----
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                break
        episode_rewards.append(episode_reward)
        print('Reward %4d; ' % (episode_reward,))
        game += 1
    env.close()


if __name__ == "__main__":
    main()
