# -*- coding: utf-8 -*-
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import numpy as np
import tensorflow as tf


def lkrelu(x, slope=0.05):
    return tf.maximum(slope * x, x)


def nature_cnn(unscaled_images):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def mlp1(unscaled_vector):
    h1 = lkrelu(fc(unscaled_vector, 'fc1-1', nh=32, init_scale=np.sqrt(1.0)))
    h2 = lkrelu(fc(h1, 'fc1-2', nh=16, init_scale=np.sqrt(1.0)))
    h3 = lkrelu(fc(h2, 'fc1-3', nh=8, init_scale=np.sqrt(1.0)))
    return h3


def mlp2(unscaled_vector):
    h1 = lkrelu(fc(unscaled_vector, 'fc2-1', nh=128, init_scale=np.sqrt(1.0)))
    h2 = lkrelu(fc(h1, 'fc2-2', nh=64, init_scale=np.sqrt(1.0)))
    h3 = lkrelu(fc(h2, 'fc2-3', nh=16, init_scale=np.sqrt(1.0)))
    return h3


def mlp3(unscaled_vector):
    h1 = lkrelu(fc(unscaled_vector, 'fc3-1', nh=256, init_scale=np.sqrt(1.0)))
    h2 = lkrelu(fc(h1, 'fc3-2', nh=128, init_scale=np.sqrt(1.0)))
    h3 = lkrelu(fc(h2, 'fc3-3', nh=64, init_scale=np.sqrt(1.0)))
    h4 = lkrelu(fc(h3, 'fc3-4', nh=32, init_scale=np.sqrt(1.0)))
    return h4


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=64, reuse=False):
        nenv = nbatch // nsteps
        ob_shape = (nbatch, ob_space.shape[0])

        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name="input")  # obs
        M = tf.placeholder(tf.float32, [nbatch], name='mask')  # mask
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2], name='cellstate')  # output+states

        activ = tf.tanh

        with tf.variable_scope("model", reuse=reuse):
            # x1, x2 = tf.split(value=X, num_or_size_splits=[17, 64], axis=1)
            # x1 = mlp1(x1)
            # x2 = mlp2(x2)
            # h = tf.concat([x1, x2], axis=1)
            h = mlp2(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            concat_layer = seq_to_batch(h5)

            "policy"
            pi_001 = lkrelu(fc(concat_layer, 'pi_001', nh=32, init_scale=np.sqrt(1.0)))
            pi_002 = lkrelu(fc(pi_001, 'pi_002', nh=16, init_scale=np.sqrt(1.0)))
            pi = activ(fc(pi_002, "pi", actdim))

            "value"
            vf_001 = lkrelu(fc(concat_layer, 'vf_001', nh=32, init_scale=np.sqrt(1.0)))
            vf_002 = lkrelu(fc(vf_001, 'vf_002', nh=16, init_scale=np.sqrt(1.0)))
            vf = fc(vf_002, 'vf', 1)

            # logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 - 2.1], axis=1)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        #a0 = pi
        a0 = self.pd.sample()

        neglogp0 = self.pd.neglogp(a0)
        self.action = tf.identity(a0, name='action')
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            action, v, sn, neglogp = sess.run([self.action, v0, snew, neglogp0], {X: ob, S: state, M: mask})
            return action, v, sn, neglogp

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = v0
        self.step = step
        self.value = value


class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states

        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]

        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        activ = tf.tanh
        with tf.variable_scope("pi", reuse=reuse):
            h1 = lkrelu(fc(X, 'pi_fc1', nh=128, init_scale=np.sqrt(2)))
            h2 = lkrelu(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            h3 = lkrelu(fc(h2, 'pi_fc3', nh=32, init_scale=np.sqrt(2)))
            h4 = lkrelu(fc(h3, 'pi_fc4', nh=16, init_scale=np.sqrt(2)))
            pi = fc(h4, 'pi', actdim, init_scale=0.01)

            h1 = lkrelu(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
            h2 = lkrelu(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            h3 = lkrelu(fc(h2, 'vf_fc3', nh=32, init_scale=np.sqrt(2)))
            h4 = lkrelu(fc(h3, 'vf_fc4', nh=16, init_scale=np.sqrt(2)))
            vf = fc(h4, 'vf', 1)[:, 0]

#             logstd = tf.get_variable(name="logstd", shape=[1, actdim],
#                                      initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 - 1.0], axis=1)

        self.pdtype = make_pdtype(ac_space)  # Probability distribution function  pd
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        # a1 = tf.clip_by_value(a0, -3, 3) / 3
        a1 = activ(a0)
        self.action = tf.identity(a1, name='action')

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            action, v, neglogp = sess.run([self.action, vf, neglogp0], {X: ob})
            return action, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
