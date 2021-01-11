# -*- coding:utf-8 -*-
import numpy as np


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class Runner(object):
    def __init__(self, *, sess, env, model, nsteps, gamma, lam):
        self.sess = sess
        self.env = env
        self.model = model
        self.env_num = env.num_envs
        self.num_env = env.num_envs
        self.obs = np.zeros((self.env_num,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(self.env_num)]
        self.count = 0
        self._write_txt = []

    def run(self):
        # initialize obs, rewards, actions and so on.
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_states = [], [], [], [], [], [], []
        epinfos = []
        r = []
        l = []
        episodes = 0
        arrived_number = 0

        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            if self.states is not None:
                mb_states.append(self.states)
            else:
                mb_states = None

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

            if self.num_env == 1:
                if self.dones:
                    r.append(infos[0]['episode']['r'])
                    l.append(infos[0]['episode']['l'])

                    episodes += 1

                    if rewards[0, 0] > 5:
                        arrived_number += 1
            else:
                for i in range(self.num_env):
                    if self.dones[i]:
                        r.append(infos[i]['episode']['r'])
                        l.append(infos[i]['episode']['l'])

                        episodes += 1

                        if rewards[i, 0] > 5:
                            arrived_number += 1

        print("buffer mean reward: %.6f   buffer mean episode length: %.2f   episodes number: %d   "
              "arrived number: %d" % (np.mean(r), np.mean(l), episodes, arrived_number))

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape([-1, self.num_env])
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape([-1, self.num_env])
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_states = np.asarray(mb_states, dtype=np.float32)
        mb_states = mb_states.reshape([-1, mb_states.shape[-1]])
        last_values = self.model.value(self.obs, self.states, self.dones)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns[:] = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos, np.mean(r), np.mean(l))
