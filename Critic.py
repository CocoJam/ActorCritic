import numpy as np
import tensorflow as tf

class Critic:
    """docstring for Critic."""
    def __init__(self, session, state_size, learning_rate=0.01):
        self.session = session
        self.state_size = tf.placeholder(tf.float32, state_size,name="state_size")
        self.expected_next_value = tf.placeholder(tf.float32, [1,1],name="esitmate_next_value")
        self.reward = tf.placeholder(tf.float32,None,name= "reward")
        GAMMA = 0.9
        with tf.variable_scope("Critic"):
            layer_1= tf.layers.dense(
             inputs=self.state_size,
             units=20,
             activation=tf.nn.relu,
             kernel_initializer=tf.random_normal_initializer(0., .1),
             bias_initializer=tf.constant_initializer(0.1),
             name="layer_1")
            self.estimated_v = tf.layers.dense(
                 inputs=layer_1,
                 units=1,
                 activation= None,
                 kernel_initializer=tf.random_normal_initializer(0., .1),
                 bias_initializer=tf.constant_initializer(0.1),
                 name='estimated_v'
             )
        with tf.variable_scope("Squared_td_error"):
            self.td_error = self.reward + GAMMA * self.expected_next_value - self.estimated_v
            tf.summary.scalar('Critic_td_error', self.td_error)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    #learn and based on the given state, next state and reward to provide value estimate, and provide TD_error based on state and next state
    def learn(self,current_state,rewards,next_state):
        current_state, next_state = current_state[np.newaxis,:], next_state[np.newaxis,:]
        feed_dict = {self.state_size: next_state}
        estimated_next_state_value= self.session.run(self.estimated_v,feed_dict)
        feed_dict = {self.state_size: current_state, self.expected_next_value: estimated_next_state_value, self.reward: rewards}
        td_error,_ = self.session.run([self.td_error,self.loss], feed_dict)
        return td_error
