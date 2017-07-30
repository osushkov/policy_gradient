
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
            tf.random_uniform([self.layer_size], minval=-init_range, maxval=init_range),
            dtype=tf.float32)

        self.layer_output = activation_func(tf.matmul(input_tensor, self.weights) + self.bias)


class Learner(LearnerInstance):
    def __init__(self, networkSpec):
        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize
        self.reward_discount = 0.99

        self.layer_sizes = [self.num_inputs, self.num_inputs / 2]
        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run([self.init_op])
            self.sess.run(self.update_ops)


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            self._buildTargetNetwork()
            self._buildLearnNetwork()
            self.init_op = tf.global_variables_initializer()


    def _buildTargetNetwork(self):
        self.target_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))

        self.target_network = []
        for ls in self.layer_sizes:
            if len(self.target_network) == 0:
                num_inputs = self.num_inputs
                input_tensor = self.target_network_input
            else:
                num_inputs = self.target_network[-1].layer_size
                input_tensor = self.target_network[-1].layer_output

            self.target_network.append(NNLayer(num_inputs, ls, tf.nn.elu, input_tensor))

        pl = self.target_network[-1]
        self.target_network.append(NNLayer(pl.layer_size, self.num_outputs, tf.nn.tanh, pl.layer_output))
        self.target_network_output = tf.stop_gradient(self.target_network[-1].layer_output)


    def _buildLearnNetwork(self):
        # learning network
        self.learn_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))
        self.learn_network_action_index = tf.placeholder(tf.int32, shape=(self.max_batch_size))
        self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
        self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size))
        self.learn_rate = tf.placeholder(tf.float32)

        self.learn_network = []
        for ls in self.layer_sizes:
            if len(self.learn_network) == 0:
                num_inputs = self.num_inputs
                input_tensor = self.learn_network_input
            else:
                num_inputs = self.learn_network[-1].layer_size
                input_tensor = self.learn_network[-1].layer_output

            self.learn_network.append(NNLayer(num_inputs, ls, tf.nn.elu, input_tensor))

        pl = self.learn_network[-1]
        self.learn_network.append(NNLayer(pl.layer_size, self.num_outputs, tf.nn.tanh, pl.layer_output))
        self.learn_network_output = self.learn_network[-1].layer_output

        terminating_target = self.learn_network_reward
        intermediate_target = self.learn_network_reward + (tf.reduce_max(self.target_network_output, axis=1) * self.reward_discount)
        self.desired_output = tf.stop_gradient(
            tf.where(self.learn_network_terminal_mask, terminating_target, intermediate_target))

        index_range = tf.constant(np.arange(self.max_batch_size), dtype=tf.int32)
        action_indices = tf.stack([index_range, self.learn_network_action_index], axis=1)
        self.indexed_output = tf.gather_nd(self.learn_network_output, action_indices)

        self.learn_loss = tf.losses.mean_squared_error(self.desired_output, self.indexed_output)

        opt = tf.train.AdamOptimizer(self.learn_rate)
        vars_to_optimise = []
        for ll in self.learn_network:
            vars_to_optimise.append(ll.weights)
            vars_to_optimise.append(ll.bias)

        self.learn_optimizer = opt.minimize(self.learn_loss, var_list=vars_to_optimise)

        self.update_ops = []
        assert (len(self.target_network) == len(self.learn_network))
        for i in range(len(self.target_network)):
            dst_weights = self.target_network[i].weights
            src_weights = self.learn_network[i].weights
            self.update_ops.append(tf.assign(dst_weights, src_weights, validate_shape=True, use_locking=True))

            dst_bias = self.target_network[i].bias
            src_bias = self.learn_network[i].bias
            self.update_ops.append(tf.assign(dst_bias, src_bias, validate_shape=True, use_locking=True))


    def Learn(self, batch):
        assert (batch.initialStates.ndim == 2 and batch.successorStates.ndim == 2)
        assert (batch.initialStates.shape[0] == self.max_batch_size and batch.initialStates.shape[1] == self.num_inputs)
        assert (batch.successorStates.shape[0] == self.max_batch_size and batch.successorStates.shape[1] == self.num_inputs)
        assert (batch.actionsTaken.ndim == 1 and batch.actionsTaken.shape[0] == self.max_batch_size)
        assert (batch.isEndStateTerminal.ndim == 1 and batch.isEndStateTerminal.shape[0] == self.max_batch_size)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.initialStates,
                self.target_network_input: batch.successorStates,
                self.learn_network_action_index: batch.actionsTaken,
                self.learn_network_terminal_mask: batch.isEndStateTerminal,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate
            }

            self.sess.run([self.learn_optimizer], feed_dict=learn_feed_dict)


    def UpdateTargetParams(self):
        with self.sess.as_default():
            self.sess.run(self.update_ops)


    def QFunction(self, state):
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
            feed_dict = {self.target_network_input: state}
            return self.sess.run([self.target_network_output], feed_dict=feed_dict)[0][:original_input_size, :]
