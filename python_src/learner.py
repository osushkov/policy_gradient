
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sonnet as snt
import math
import random
import sys


class Learner(LearnerInstance):
    def __init__(self, networkSpec):
        self.action_values_avrg = [0.0] * networkSpec.numOutputs

        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize

        self.layer_sizes = [self.num_inputs, self.num_inputs / 2, self.num_inputs / 4]
        self.value_layer_sizes = [self.num_inputs, self.num_inputs / 2, self.num_inputs / 4]

        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run([self.init_op])


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.learn_network_input = tf.placeholder(tf.float32,
                                                      shape=(self.max_batch_size, self.num_inputs))
            self.learn_network_action_index = tf.placeholder(tf.int32, shape=(self.max_batch_size))
            self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
            self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size))
            self.learn_rate = tf.placeholder(tf.float32)

            self._buildValueNetwork(self.learn_network_input)
            self._buildLearnNetwork(self.learn_network_input)

            self.init_op = tf.global_variables_initializer()


    def _buildLearnNetwork(self, network_input):
        prev_output = network_input
        for ls in self.layer_sizes:
            layer = snt.Linear(ls)
            prev_output = tf.nn.relu(layer(prev_output))

        output_layer = snt.Linear(self.num_outputs)
        self.learn_network_output = tf.nn.softmax(output_layer(prev_output))

        index_range = tf.constant(np.arange(self.max_batch_size), dtype=tf.int32)
        action_indices = tf.stack([index_range, self.learn_network_action_index], axis=1)
        self.indexed_output = tf.gather_nd(self.learn_network_output, action_indices)

        advantage = tf.stop_gradient(self.learn_network_reward - self._value_network_output)
        self.log_output = tf.log(self.indexed_output + 0.000001)
        self.learn_loss = tf.reduce_mean(advantage * tf.negative(self.log_output))

        opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        self.learn_optimizer = opt.minimize(self.learn_loss)

    def _buildValueNetwork(self, network_input):
        prev_output = network_input
        for ls in self.value_layer_sizes:
            layer = snt.Linear(ls)
            prev_output = tf.nn.relu(layer(prev_output))

        output_layer = snt.Linear(1)
        self._value_network_output = tf.nn.tanh(output_layer(prev_output))

        loss = tf.losses.mean_squared_error(self._value_network_output,
                                            tf.reshape(self.learn_network_reward, (-1, 1)))

        opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        self.value_optimizer = opt.minimize(loss)


    def Learn(self, batch):
        assert (batch.initialStates.ndim == 2)
        assert (batch.initialStates.shape[0] == self.max_batch_size and batch.initialStates.shape[1] == self.num_inputs)
        assert (batch.actionsTaken.ndim == 1 and batch.actionsTaken.shape[0] == self.max_batch_size)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        alpha = 0.001
        for i in range(batch.actionsTaken.shape[0]):
            self.action_values_avrg[batch.actionsTaken[i]] = (1.0 - alpha) * self.action_values_avrg[batch.actionsTaken[i]] + alpha * batch.rewardsGained[i]

        # if random.randint(0, 100) == 0:
        #     print("action values: {}".format(self.action_values_avrg))
        #     # print("boards:\n{}".format(batch.initialStates.reshape(-1, 6, 14)))
        #     print("actions:\n{}".format(batch.actionsTaken))
        #     print("rewards:\n{}".format(batch.rewardsGained))

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.initialStates,
                self.learn_network_action_index: batch.actionsTaken,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate,
            }

            _ = self.sess.run(self.learn_optimizer, feed_dict=learn_feed_dict)
            # _, ll, io, no, lo = self.sess.run([self.learn_optimizer, self.learn_loss, self.indexed_output, self.learn_network_output, self.log_output], feed_dict=learn_feed_dict)
            # if random.randint(0, 1000) == 0:
            #     print("learn loss: {}".format(ll))
            #
            # if math.isnan(ll):
            #     print("action values: {}".format(self.action_values_avrg))
            #     # print("boards:\n{}".format(batch.initialStates))
            #     print("actions:\n{}".format(batch.actionsTaken))
            #     print("rewards:\n{}".format(batch.rewardsGained))
            #     print("indexed output:\n{}".format(io))
            #     print("network out:\n{}".format(no))
            #
            #     print("prev loss: {}".format(self.prev_ll))
            #     print("prev action values: {}".format(self.prev_action_values_avrg))
            #     # print("boards:\n{}".format(batch.initialStates))
            #     print("prev actions:\n{}".format(self.prev_batch.actionsTaken))
            #     print("prev rewards:\n{}".format(self.prev_batch.rewardsGained))
            #     print("prev indexed output:\n{}".format(self.prev_io))
            #     print("prev network out:\n{}".format(self.prev_no))
            #     print("prev log output:\n{}".format(self.prev_lo))
            #     sys.exit()
            #
            # self.prev_action_values_avrg = self.action_values_avrg
            # self.prev_batch = batch
            # self.prev_io = io
            # self.prev_no = no
            # self.prev_ll = ll
            # self.prev_lo = lo


    def LearnValue(self, batch):
        assert (batch.states.ndim == 2)
        assert (batch.states.shape[0] == self.max_batch_size and batch.states.shape[1] == self.num_inputs)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.states,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate,
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

            output = self.sess.run([self.learn_network_output], feed_dict=feed_dict)[0][:original_input_size, :]
            if random.randint(0, 1000) == 0 and False:
                np.set_printoptions(precision=2)
                print(output[0])
            return output
