import gym
import tensorflow as tf
import numpy as np
from Actor import Actor
from Critic import Critic

env= gym.make("CartPole-v0")
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

Max_itieration = 3000
Max_steps = 100
learning_rate_actor = 0.001
learning_rate_critic = 0.01
DISPLAY_REWARD_THRESHOLD = 200
RENDER =False
session = tf.Session()

actor = Actor(session, state_size= [1,N_F], action_size= N_A, learning_rate = learning_rate_actor)
critic = Critic(session, state_size= [1,N_F], learning_rate = learning_rate_critic)

session.run(tf.global_variables_initializer())

for ep in range(Max_itieration):
    env_state = env.reset()
    time_step = 0
    tracking_rewards = []
    while True:
        if RENDER: env.render()
        action = actor.action_selection(env_state)
        env_next_state, rewards, done, information = env.step(action)
        time_step += 1
        # if done:
        #     rewards= -20
        tracking_rewards.append(rewards)
        td_error = critic.learn(env_state, rewards,env_next_state)
        actor.learn(env_state,action,td_error)
        env_state = env_next_state

        if done or time_step >= Max_steps:
            total_reward = sum(tracking_rewards)
            if 'running_reward' not in globals():
                running_reward = total_reward
            else:
                running_reward = running_reward * 0.95 + total_reward * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            tf.summary.scalar('Total_reward', running_reward)
            print("episode:", ep, "  reward:", int(running_reward))
            break


merged = tf.summary.merge_all()
tf.summary.FileWriter("logs/", sess.graph)
