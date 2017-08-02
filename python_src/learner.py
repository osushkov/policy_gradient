
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


class NNLayer:
    def __init__(self, num_inputs, layer_size, activation_func, input_tensor):
        self.num_inputs = num_inputs
        self.layer_size = layer_size

        init_range = 1.0 / self.num_inputs

        self.weights = tf.Variable(
            tf.random_uniform([self.num_inputs, self.layer_size], minval=-init_range, maxval=init_range),
            dtype=tf.float32)

        self.bias = tf.Variable(
            tf.zeros((self.layer_size)), dtype=tf.float32)

        self.layer_output = activation_func(tf.matmul(input_tensor, self.weights) + self.bias)


class Learner(LearnerInstance):
    def __init__(self, networkSpec):
        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize

        self.layer_sizes = [self.num_inputs, self.num_inputs / 2]
        self.value_layer_sizes = [self.num_inputs, self.num_inputs / 2]

        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run([self.init_op])


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.learn_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))
            self.learn_network_action_index = tf.placeholder(tf.int32, shape=(self.max_batch_size))
            self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
            self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size))
            self.learn_rate = tf.placeholder(tf.float32)

            self._buildValueNetwork()
            self._buildLearnNetwork()

            self.init_op = tf.global_variables_initializer()


    def _buildLearnNetwork(self):
        self.learn_network = []
        for ls in self.layer_sizes:
            if len(self.learn_network) == 0:
                num_inputs = self.num_inputs
                input_tensor = self.learn_network_input
            else:
                num_inputs = self.learn_network[-1].layer_size
                input_tensor = self.learn_network[-1].layer_output

            self.learn_network.append(NNLayer(num_inputs, ls, tf.nn.tanh, input_tensor))

        pl = self.learn_network[-1]
        self.learn_network.append(NNLayer(pl.layer_size, self.num_outputs, tf.identity, pl.layer_output))
        self.learn_network_output = self.learn_network[-1].layer_output

        # index_range = tf.constant(np.arange(self.max_batch_size), dtype=tf.int32)
        # action_indices = tf.stack([index_range, self.learn_network_action_index], axis=1)
        # indexed_output = tf.gather_nd(self.learn_network_output, action_indices)
        #
        # advantage = tf.stop_gradient(self.learn_network_reward - self._value_network_output)
        # loss = tf.negative(tf.reduce_mean(advantage * tf.log(indexed_output)))

        advantage = tf.stop_gradient(self.learn_network_reward - self._value_network_output)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.learn_network_output, labels=self.learn_network_action_index)
        self.learn_loss = tf.reduce_mean(loss * advantage)
        # average_reward = tf.reduce_mean(self.learn_network_reward)

        opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        self.learn_optimizer = opt.minimize(self.learn_loss)

    def _buildValueNetwork(self):
        self._value_network = []

        for ls in self.value_layer_sizes:
            if len(self._value_network) == 0:
                num_inputs = self.num_inputs
                input_tensor = self.learn_network_input
            else:
                num_inputs = self._value_network[-1].layer_size
                input_tensor = self._value_network[-1].layer_output

            self._value_network.append(NNLayer(num_inputs, ls, tf.nn.tanh, input_tensor))

        pl = self._value_network[-1]
        self._value_network.append(NNLayer(pl.layer_size, 1, tf.nn.tanh, pl.layer_output))
        self._value_network_output = self._value_network[-1].layer_output

        loss = tf.losses.mean_squared_error(self._value_network_output, tf.reshape(self.learn_network_reward, (-1, 1)))

        opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        self.value_optimizer = opt.minimize(loss)


    def Learn(self, batch):
        assert (batch.initialStates.ndim == 2)
        assert (batch.initialStates.shape[0] == self.max_batch_size and batch.initialStates.shape[1] == self.num_inputs)
        assert (batch.actionsTaken.ndim == 1 and batch.actionsTaken.shape[0] == self.max_batch_size)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        # print("boards:\n{}".format(batch.initialStates.reshape(-1, 6, 14)))
        # print("actions:\n{}".format(batch.actionsTaken))
        # print("rewards:\n{}".format(batch.rewardsGained))
        # print("lr: {}".format(batch.learnRate))

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.initialStates,
                self.learn_network_action_index: batch.actionsTaken,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate
            }

            _, ll = self.sess.run([self.learn_optimizer, self.learn_loss], feed_dict=learn_feed_dict)
            # print("learn loss: {}".format(ll))

    def LearnValue(self, batch):
        assert (batch.states.ndim == 2)
        assert (batch.states.shape[0] == self.max_batch_size and batch.states.shape[1] == self.num_inputs)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.states,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate
            }

            _ = self.sess.run(self.value_optimizer, feed_dict=learn_feed_dict)

    def PolicyFunction(self, state):
        assert(state.ndim == 1 or state.ndim == 2)

        if state.ndim == 1:
            assert(state.shape[0] == self.num_inputs)
            state = state.reshape(1, self.num_inputs)

        if state.shape[0] > self.max_batch_size or state.shape[1] != self.num_inputs:
            raise Exception("Invalid state, wrong shape")

        original_input_size = state.shape[0]
        if state.shape[0] < self.max_batch_size:
            padded_state = np.zeros((self.max_batch_size, self.num_inputs), dtype=np.float32)
            padded_state[:state.shape[0], :] = state
            state = padded_state

        assert (state.shape[0] == self.max_batch_size and state.shape[1] == self.num_inputs)

        with self.sess.as_default():
            feed_dict = {
                self.learn_network_input: state,
            }
            return self.sess.run([self.learn_network_output], feed_dict=feed_dict)[0][:original_input_size, :]
