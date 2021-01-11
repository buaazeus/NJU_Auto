# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
from collections import deque
from baselines.common import set_global_seeds
from alg.logger import MyLogger
from alg.models import Model
from alg.runner import Runner

#一个函数，返回值为固定值val
def constfn(val):
    def f(_):
        return val
    return f


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, model_path, log_path,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          nminibatches=4, noptepochs=4, cliprange=0.2,
          seed=0):
    """Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Args:
        policy:                           policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies
        env: baselines.common.vec_env.VecEnv
                                          environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.
        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)
        total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)
        ent_coef: float                   policy entropy coefficient in the optimization objective
        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.
        model_path: str                   path of saved model.
        log_path: str                     path of log.
        vf_coef: float                    value function loss coefficient in the optimization objective
        max_grad_norm: float or None      gradient norm clipping coefficient
        gamma: float                      discounting factor
        lam: float                        advantage estimation discounting factor (lambda in the paper)
        nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                          should be smaller or equal than number of environments run in parallel.
        noptepochs: int                   number of training epochs per update
        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training
        seed: int                         seed of env.
    """

    set_global_seeds(seed)
    mylogger = MyLogger(log_path)

    if isinstance(lr, float):
        lr = constfn(lr)  # get the learning rate
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)  # get the clip range
    else:
        assert callable(cliprange)

    total_timesteps = int(total_timesteps)  # set the total_timesteps for train
    nenvs = env.num_envs  # number of the env
    ob_space = env.observation_space  # ob space from nenvs
    ac_space = env.action_space  # action space
    # Calculate the batch_size
    nbatch = nenvs * nsteps # size of buffer

    lstm_steps = 32
    lstm_nenvs = nbatch // lstm_steps
    nbatch_train = nbatch // nminibatches
    max_step = total_timesteps // nbatch * nminibatches * noptepochs

    # define policy/value model
    sess = tf.Session()
    make_model = lambda: Model(sess=sess, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               steps=lstm_steps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, max_step=max_step)
    model = make_model()

    # load saved model
    saver = tf.train.Saver(max_to_keep=10000)
    sess.run(tf.global_variables_initializer())
    #如果从头开始训练，直接删除checkpoint文件
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded: ", checkpoint.model_checkpoint_path)
    else:
        print("Could not find saved model.")

    runner = Runner(sess=sess, env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    mylogger.add_sess_graph(sess.graph)
    epinfobuf = deque(maxlen=100)
    nupdates = total_timesteps // nbatch
    try:
        for update in range(1, nupdates + 1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)
            start_time = time.time()
            obs, returns, masks, actions, values, neglogpacs, states, epinfos, mean_r, mean_l = runner.run()
            print('step time: %.2f' % (time.time() - start_time))
            epinfobuf.extend(epinfos)
            mblossvals = []
            if states is None:  # non-recurrent version
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        train_rtn = model.train(lrnow, cliprangenow, *slices)
                        mblossvals.append(train_rtn)
                        global_step = train_rtn[-1]

                        mylogger.write_summary_scalar(global_step, "loss", train_rtn[0])
                        mylogger.write_summary_scalar(global_step, "pg_loss", train_rtn[1])
                        mylogger.write_summary_scalar(global_step, "vf_loss", train_rtn[2])
                        mylogger.write_summary_scalar(global_step, "entropy", train_rtn[3])
                        mylogger.write_summary_scalar(global_step,"learning_rate",train_rtn[5])
            else:
                assert lstm_nenvs % nminibatches == 0
                envinds = np.arange(lstm_nenvs)
                flatinds = np.arange(lstm_nenvs * lstm_steps).reshape(lstm_nenvs, lstm_steps)
                envsperbatch = nbatch_train // lstm_steps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, lstm_nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        train_rtn = model.train(lrnow, cliprangenow, *slices, mbstates)
                        mblossvals.append(train_rtn)
                        global_step = train_rtn[-1]
                        mylogger.write_summary_scalar(global_step, "loss", train_rtn[0])
                        mylogger.write_summary_scalar(global_step, "pg_loss", train_rtn[1])
                        mylogger.write_summary_scalar(global_step, "vf_loss", train_rtn[2])
                        mylogger.write_summary_scalar(global_step, "entropy", train_rtn[3])
                        mylogger.write_summary_scalar(global_step, "learning_rate", train_rtn[5])

            lossvals = np.mean(mblossvals, axis=0)
            number = int(global_step / (nminibatches * noptepochs))
            mylogger.write_summary_scalar(number, "mean_return", mean_r)
            mylogger.write_summary_scalar(number, "mean_length", mean_l)

            print("%d / %d   %d   lr: %.6f   ent_coef: %.6f   vf_coef: %.6f   loss: %.6f   policy_loss: %.6f   "
                  "value_loss: %.6f   policy_entropy: %.6f\n"
                  % (number, nupdates, global_step, lossvals[5], lossvals[4], vf_coef, lossvals[0], lossvals[1],
                     lossvals[2], lossvals[3]))

            # save model.
            if update % 1 == 0:
                saver.save(sess, model_path, global_step=global_step)
                print("Model has been saved successfully!\n")

            if number > nupdates:
                print("end!!!")
                break
        env.close()

    except KeyboardInterrupt:
        env.close()

