# -*- coding:utf-8 -*-

import tensorflow as tf


class ImportLstmGraph(object):
    """  Importing and running isolated LSTM graph
    """
    def __init__(self, loc):
        """Create local graph and use it in the session.

        Args:
            loc: path of local graph
        """
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.X = self.graph.get_tensor_by_name('input:0')
            self.S = self.graph.get_tensor_by_name('cellstate:0')
            self.M = self.graph.get_tensor_by_name('mask:0')
            self.action = self.graph.get_tensor_by_name('action:0')

    def policy(self, ob, state, mask):
        actions, states = self.sess.run([self.action, self.S], feed_dict={self.X: ob, self.S: state, self.M: mask})
        return actions, states

    def close_sess(self):
        self.sess.close()
