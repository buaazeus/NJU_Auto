# -*- coding: utf-8 -*-
import joblib
import tensorflow as tf

"""定义PPO更新梯度的类，主要定义了loss functions、ppo需要的各种变量的placeholder以及优化方法等"""
class Model(object):
    def __init__(self, *, sess, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 steps, ent_coef, vf_coef, max_grad_norm, max_step):
        # 用于将一条序列中的每一步ob送入policy获得action（用来采样action）
        # nbatch_act就是Unity的真实环境数，有几个环境就送几个单步状态到policy中
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=tf.AUTO_REUSE)
        # 将一个batch的数据送到policy网络中去更新lstm的参数
        # 这里的nbatch_train就是真实的batchsize即buffer/minibatch
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps=steps, reuse=tf.AUTO_REUSE)

        A = train_model.pdtype.sample_placeholder([None])  # action
        ADV = tf.placeholder(tf.float32, [None])  # advantage
        R = tf.placeholder(tf.float32, [None])  # return
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])  # old -logp(action)
        OLDVPRED = tf.placeholder(tf.float32, [None])  # old value prediction
        LR = tf.placeholder(tf.float32, [])  # learning rate
        CLIPRANGE = tf.placeholder(tf.float32, [])  # clip range
        neglogpac = train_model.pd.neglogp(A)  # -logp(action)
        entropy = tf.reduce_mean(train_model.pd.entropy())
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # print("R = ", tf.identity(R))
        vf_losses1 = tf.square(vpred - R)
        # print("vf_losses1 = ", tf.identity(vf_losses1))
        vf_losses2 = tf.square(vpredclipped - R)
        # print("vf_losses2 = ", tf.identity(vf_losses2))

        # vf_loss理论上也应该下降，但实际上是波动的
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # print("vf_loss = ", tf.identity(vf_loss))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        # 论文中是"ADV * ratio"和"ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)"取小的为了pgloss
        # 这里用了负数，就是取大
        # 原本是最大化pgloss，但是这里是取了负数就是最小化这里的pgloss
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # pgloss是应该要下降的
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        # 用于使用KL散度的PPO算法，这里用不到
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # 定义全局的执行步数
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # 多项式递减学习率和entropy
        # decay_ent_coef = tf.train.polynomial_decay(ent_coef, self.global_step, max_step//5000, 1e-5, power=1.0,
        # cycle=True)
        # learning_rate = tf.train.polynomial_decay(LR, self.global_step, max_step//5000, 1e-6, power=1.0, cycle=True)
        decay_ent_coef = tf.train.polynomial_decay(ent_coef, self.global_step, max_step, 1e-5, power=1.0, cycle=True)
        learning_rate = tf.train.polynomial_decay(LR, self.global_step, max_step, 1e-6, power=1.0, cycle=True)

        '''This objective can further be augmented by adding an entropy bonus to ensure suﬃcient exploration'''
        # 整体loss要减
        loss = pg_loss - entropy * decay_ent_coef + vf_loss * vf_coef

        # 要学习更新的参数集合
        with tf.variable_scope('model'):
            params = tf.trainable_variables()

        # 求loss关于params的梯度
        # 用梯度更新网络
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        _train = trainer.apply_gradients(grads, global_step=self.global_step, name="apply_gradients")

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # 从runner中返回的是return = adv+reward
            advs = returns - values
            # 对adv做归一
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # 定义要喂进policy网络的数据字典
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}

            '''for lstm/rnn'''
            if states is not None:
                """test"""
                # print("states is not None and shape is:"+str(states.shape))
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            # 用loss更新网络
            _loss, _gp_loss, _vf_loss, _entropy, _decay_ent_coef, _learning_rate = sess.run(
                [loss, pg_loss, vf_loss, entropy, decay_ent_coef, learning_rate, _train], td_map)[:-1]

            _global_step = sess.run(self.global_step)

            return [_loss, _gp_loss, _vf_loss, _entropy, _decay_ent_coef, _learning_rate, _global_step]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            """test"""
            print("This is models save")
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            """test"""
            print("This is models load")
            saver = tf.train.import_meta_graph('model/ckpt-0.meta')
            model_file = tf.train.latest_checkpoint('model/')
            saver.restore(sess, model_file)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        sess.run(tf.global_variables_initializer())  # pylint: disable=E1101
