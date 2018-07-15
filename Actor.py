import tensorflow as tf
import numpy as np
class Actor():
    def __init__(self, session, state_size, action_size, learning_rate=0.001):
        self.session= session
        self.state_size = tf.placeholder(tf.float32, state_size,name="states_size")
        self.action_size = tf.placeholder(tf.int32,None,name="action_size")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        with tf.variable_scope("Actor"):
            layer_1= tf.layers.dense(
             inputs=self.state_size,
             units=20,
             activation=tf.nn.relu,
             kernel_initializer=tf.random_normal_initializer(0., .1),
             bias_initializer=tf.constant_initializer(0.1),
             name="layer_1")



            self.actions_probabilities = tf.layers.dense(
                inputs=layer_1,
                units=action_size,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='actions_probabilities'
            )
        with tf.variable_scope("expected_value"):
            log_probability= tf.log(self.actions_probabilities[0,self.action_size])
            self.expected_value = tf.reduce_mean(log_probability * self.td_error)
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.expected_value)

# Learning based on the Critic's td_error, and based on the current state of actor and action taking.
    def learn(self,current_state,current_action,td_error):
        current_state = current_state[np.newaxis,:]
        feed_dict = {self.state_size: current_state,self.action_size: current_action,self.td_error: td_error}
        _,expected_value = self.session.run([self.train_op,self.expected_value],feed_dict)
        return expected_value

# Action selected based on the current state, which is not the action state included.
    def action_selection(self,current_state):
        current_state = current_state[np.newaxis,:]
        feed_dict = {self.state_size: current_state}
        actions_probabilities = self.session.run(self.actions_probabilities, feed_dict)
        return np.random.choice(np.arange(actions_probabilities.shape[1]), p=actions_probabilities.ravel())
