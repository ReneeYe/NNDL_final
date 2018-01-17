import os
import sys
from collections import deque
import numpy as np
import tensorflow as tf

class DQNAgent:
    """
    Multi Layer CNN with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, layers, rows, cols, model_dir):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions.tolist()
        self.n_actions = len(self.enable_actions)
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self.minibatch_size = 128
        self.replay_memory_size = 10000
        self.learning_rate = 0.001
        # self.learning_rate = 0.005
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = model_dir
        # self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.keep_prob_rate = 0.7 # dropout 0.3
        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0

    def init_model(self):
        # input layer (rows x cols)
        self.x = tf.placeholder(tf.float32, [None, self.rows, self.cols, self.layers])

        x_image = tf.reshape(self.x,[-1,8,8,3])
        # conv1: patch 2*2, in 3, out 6
        w_conv1 = tf.Variable(tf.truncated_normal([2,2,3,6],stddev = 0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape = [6]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides = [1,1,1,1], padding= "SAME") + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides =[1,2,2,1], padding = "SAME")

        # conv2: patch 2*2, in: 6, out: 12
        w_conv2 = tf.Variable(tf.truncated_normal([2,2,6,12], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape = [12]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*12])

        # FC1: 12*64 - 4*64
        W_fc1 = tf.Variable(tf.truncated_normal([48, 64], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.1, shape = [64]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = self.keep_prob_rate)

        # FC2: ie. output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([64, self.n_actions], stddev=0.01))
        b_out_init = tf.zeros([self.n_actions])
        b_out_init = b_out_init + np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0])
        b_out = tf.Variable(b_out_init)
        # b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc1_drop, W_out) + b_out
        # self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_out) + b_out)

        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def select_action(self, state, targets, epsilon):

        if np.random.rand() > epsilon:
            # random
            return np.random.choice(targets)
        else:
            # max_action Q(state, action)
            qvalue, action = self.select_enable_action(state, targets)
            return action

    def select_enable_action(self, state, targets):
        Qs = self.Q_values(state)
        # print(Qs)
        # descend = np.sort(Qs)
        index = np.argsort(Qs)
        for action in reversed(index):
            if action in targets:
                break
        # max_action Q(state, action)
        qvalue = Qs[action]

        return qvalue, action

    def store_experience(self, state, targets, action, reward, state_1, targets_1, terminal):
        # print(self.D)
        self.D.append((state, targets, action, reward, state_1, targets_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        # print(self.D)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, targets_j, action_j, reward_j, state_j_1, targets_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                qvalue, action = self.select_enable_action(state_j_1, targets_j_1)
                y_j[action_j_index] = reward_j + self.discount_factor * qvalue

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self, epoch):
        """
        epoch = 目前阶段数
        """
        model_name_iter = self.model_dir + self.environment_name + str(epoch) + ".ckpt"
        self.saver.save(self.sess, model_name_iter)

        # model_name_iter = self.environment_name + str(epoch) + ".ckpt"
        # self.saver.save(self.sess, os.path.join(self.model_dir, model_name_iter))