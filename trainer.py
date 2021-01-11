# -*- coding: utf-8 -*-

from alg import ppo2
from alg import policies
from alg.env import MakeEnv, MakeMultiEnvs


def train_ppo2(env, num_timesteps, seed, nenv, dir):
    """
    :param env:
    :param num_timesteps:
    :param seed:
    :param nenv:
    :param dir:
    :return:
    """
    ppo2.learn(policy=policies.LnLstmPolicy,  # LnLstmPolicy
               model_path='model/' + dir,
               log_path='log/' + dir,
               env=env,
               nsteps=int(2 ** 16 / nenv),  # int(2 ** 17/nenv)
               total_timesteps=num_timesteps,
               ent_coef=0.5,
               vf_coef=1.0,
               lr=0.0005,
               max_grad_norm=0.5,
               gamma=0.99,
               lam=0.99,
               nminibatches=32,
               noptepochs=16,  # epochs
               cliprange=0.2,
               seed=seed)


def main():
    """
    :return:
    """
    num_envs = 1

    dir = "HFReal/"
    env_path = "linux/" + dir + dir[:-1] + ".x86_64"
    env_path = None

    if num_envs == 1:
        """for test"""
        env = MakeEnv(env_path=env_path, train_model=False, no_graphics=False)  # test
    elif num_envs > 1:
        """for train"""
        env = MakeMultiEnvs(env_path=env_path, num_env=num_envs, train_model=True,
                            no_graphics=True, start_index=1)

    num_timesteps = 2e9
    train_ppo2(env, nenv=num_envs, num_timesteps=num_timesteps, dir=dir,  # Continuous_ppo2_human_dis20_v6_test012
               seed=0)


if __name__ == '__main__':
    main()
