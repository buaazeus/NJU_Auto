# -*- coding: utf-8 -*-

import numpy as np
import gym
import os
import argparse
import yaml
from gym import spaces
from mlagents.envs import UnityEnvironment
from mlagents.envs import UnityException

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench import Monitor
from baselines import logger


class UnityEnv(gym.Env):
    def __init__(self, env_directory, worker_id, train_model=True, no_graphics=False):
        """Unity environment

        Args:
            env_directory:
            worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
            train_model:
            no_graphics:
        """
        self.env = UnityEnvironment(file_name=env_directory, worker_id=worker_id, seed=1,
                                    no_graphics=no_graphics)
        self.train_model = train_model
        self.num_envs = 1
        self.envs = 1

        """get brain info"""
        self.brain_name = self.env.external_brain_names[0]
        brain = self.env.brains[self.brain_name]

        self.act_dim = brain.vector_action_space_size[0]
        self.ob_dim = brain.vector_observation_space_size

        """set action/ob space"""
        self.action_space = spaces.Box(low=np.zeros([self.act_dim]) - 1, high=np.zeros([self.act_dim]) + 1, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros([self.ob_dim]) - 1, high=np.zeros([self.ob_dim]) + 1, dtype=np.float32)

    def step(self, a):
        action = {}
        infos = {}
        action[self.brain_name] = a  # not a[0]
        info = self.env.step(vector_action=action)
        brainInfo = info[self.brain_name]
        reward = np.array(brainInfo.rewards)
        done = np.array(brainInfo.local_done)
        ob = np.array(brainInfo.vector_observations)
        return ob[-1, 0:self.ob_dim], reward, done[0], infos

    def reset(self):
        info = self.env.reset(train_mode=self.train_model)
        brainInfo = info[self.brain_name]
        ob = np.array(brainInfo.vector_observations)
        return ob[-1, 0:self.ob_dim]

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()


def MakeMultiEnvs(env_path, num_env, train_model, no_graphics, start_index=1):
    def make_env(rank):
        def _thunk():
            env = UnityEnv(env_path, rank, train_model=train_model, no_graphics=no_graphics)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def MakeEnv(env_path, train_model, no_graphics, start_index=0):
    def make_env(rank):
        def _thunk():
            env = UnityEnv(env_path, rank, train_model=train_model, no_graphics=no_graphics)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(1)])


def get_para():
    """Get parameters from the command line

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-env_path", "--env_path", help="env path")
    parser.add_argument("-model_path", "--model_path", help="model save path")
    parser.add_argument("-env_num", "--env_num", help="env num")
    parser.add_argument("-train_model", "--train_model", help="train model")
    parser.add_argument("-no_graphics", "--no_graphics", help="no graphics")

    args = parser.parse_args()

    param_keys = ['env_path', 'model_path', 'env_num', 'train_model', 'no_graphics']
    paras = {}

    paras[param_keys[0]] = args.env_path
    paras[param_keys[1]] = args.model_path
    paras[param_keys[2]] = int(args.env_num)

    if args.train_model is None:
        paras[param_keys[3]] = None
    elif args.train_model.lower() == "true":
        paras[param_keys[3]] = True  # args.train_model:
    else:
        paras[param_keys[3]] = False

    if args.no_graphics is None:
        paras[param_keys[4]] = None
    elif args.no_graphics.lower() == "true":
        paras[param_keys[4]] = True  # args.train_model:
    else:
        paras[param_keys[4]] = False
    return paras


def _load_yaml(trainer_config_path):
    try:
        with open(trainer_config_path) as data_file:
            trainer_config = yaml.load(data_file)
            return trainer_config
    except IOError:
        raise UnityException('Parameter file could not be found '
                             'at {}.'
                             .format(trainer_config_path))
    except UnicodeDecodeError:
        raise UnityException('There was an error decoding '
                             'Trainer Config from this path : {}'
                             .format(trainer_config_path))
