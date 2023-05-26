from Qlearning import Agent
import numpy as np
import tensorflow as tf


class DQN(tf.keras.Model):
    def __init__(self, n_states):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(24, activation='relu')
        self.fc3 = tf.keras.layers.Dense(n_states)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.fc3(x)

class DQN_Agent:

    def __init__(self, actions, gamma, lr=0.001):
        self.actions = actions
        self.gamma = gamma
        n_states = len(actions)
        self.model = DQN(n_states)
        self.target_model = DQN(n_states)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.lr = lr


    def act(self, state, time):
        epsilon = 0.1 ** (4*time / 500_000)
        random_action = np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        greedy_action = np.argmax(q_values[0])
        return np.random.choice([random_action, greedy_action], p = [epsilon, 1 - epsilon])
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
