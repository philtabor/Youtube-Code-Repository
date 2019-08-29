import os
import tensorflow as tf
import numpy as np

class PolicyGradientAgent():
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 chkpt_dir='tmp/checkpoints'):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.input_dims = input_dims
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'policy_network.ckpt')

    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.input_dims], name='input')
            self.label = tf.placeholder(tf.int32,
                                        shape=[None, ], name='label')
            self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

        with tf.variable_scope('layer1'):
            l1 = tf.layers.dense(inputs=self.input, units=self.layer1_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('layer2'):
            l2 = tf.layers.dense(inputs=l1, units=self.layer2_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('layer3'):
            l3 = tf.layers.dense(inputs=l2, units=self.n_actions,
                                 activation=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actions = tf.nn.softmax(l3, name='actions')

        with tf.variable_scope('loss'):
            negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=l3, labels=self.label)

            loss = negative_log_probability * self.G

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})[0]
        action = np.random.choice(self.action_space, p = probabilities )

        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        _ = self.sess.run(self.train_op,
                            feed_dict={self.input: state_memory,
                                       self.label: action_memory,
                                       self.G: G})
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        #print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)
