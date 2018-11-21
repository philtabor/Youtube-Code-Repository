import tensorflow as tf
import numpy as np

class PolicyGradientAgent():
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=4, 
                 layer1_size=16, layer2_size=16, input_dims=128):        
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
    
    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32, 
                                        shape=[None, self.input_dims], name='input')
            self.label = tf.placeholder(tf.int32, 
                                        shape=[None, ], name='label')
            self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

        with tf.variable_scope('layer1'):
            w1 = tf.get_variable('W1', [self.input_dims, self.layer1_size], 
                                 initializer=tf.initializers.random_normal(0., 0.5))
            b1 = tf.get_variable('B1', initializer=tf.constant(0.1))                                 
            l1 = tf.nn.relu(tf.matmul(self.input, w1) + b1)

        with tf.variable_scope('layer2'):
            w2 = tf.get_variable('W2', [ self.layer1_size, self.layer2_size],
                                 initializer=tf.initializers.random_normal(0., 0.5))
            b2 = tf.get_variable('B2', initializer=tf.constant(0.1))
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

        with tf.variable_scope('layer3'):
            w3 = tf.get_variable('W3', [self.layer2_size, self.n_actions],
                                 initializer=tf.initializers.random_normal(0., 0.5))
            b3 = tf.get_variable('B3', initializer=tf.constant(0.1))
            l3 = tf.matmul(l2, w3) + b3
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
        
        values = np.zeros_like(self.reward_memory)
        G = 0
        for t in reversed(range(reward_memory.shape[0])):
            G = self.gamma*G + reward_memory[t]
            values[t] = G

        mean = np.mean(values)
        std = np.std(values) if np.std(values) > 0 else 1
        values = (values - mean) / std

        _ = self.sess.run(self.train_op,
                            feed_dict={self.input: state_memory,
                                       self.label: action_memory,
                                       self.G: values})

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []