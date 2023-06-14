from classes.action import Action
from collections import deque
import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:
    """ Deep Q-Network agent in this Multi-Agent Reinforcement Learning setting.  """

    def __init__(self, actions, state_size=2, action_size=50, bufferLength=5_000, gamma=0.9, learning_rate=0.001, update_rate=1000):
        """
        Initialize a DQN agent.

        Args:
            actions (list): List of possible actions.
            state_size (int): Dimensionality of the state.
            action_size (int): Number of possible actions.
            bufferLength (int): Maximum size of the replay buffer.
            gamma (float): Discount factor for future rewards.
            learning_rate (float): Learning rate (alpha).
            update_rate (int): Copy rate target network from local network.
        """
        self.state_size = state_size  # q and theta
        self.action_size = action_size  # |A(s)| = 50
        self.create_dicts(actions)
        random.seed(np.random.randint(1, 31))

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=bufferLength)

        # Initialize hyperparameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_rate = update_rate

        # Initialize Neural Nets
        self.local_network = self.build_neural_network()
        self.target_network = self.build_neural_network()

    def create_dicts(self, actions):
        """
        Creates dictionaries for mapping actions to integers and vice versa.

        Args:
            actions (list): List of possible actions.
        """
        self.action_to_int = dict()
        self.int_to_action = dict()
        for idx, action in enumerate(actions):
            self.action_to_int[action] = idx
            self.int_to_action[idx] = action

    def build_neural_network(self):
        """
        Builds the neural network model.

        Returns:
            model (Sequential): Compiled neural network model.
        """
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        """
        Updates the target network by copying the weights from the local network.
        """
        self.target_network.set_weights(self.local_network.get_weights())

    def save_experience(self, state, action, reward, next_state, next_reward):
        """
        Saves the experience tuple to the replay buffer.

        Args:
            state (State): Current state.
            action (Action): Action taken.
            reward (float): Reward received.
            next_state (State): Next state.
            next_reward (float): Reward for the next state.
        """
        state = np.array([state.q, state.theta])
        action = np.array([action.q, action.theta])
        next_state = np.array([next_state.q, next_state.theta])
        self.replay_buffer.append((state, action, reward, next_state, next_reward))

    def get_batch(self, batch_size):
        """
        Randomly samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Size of the batch to sample.

        Returns:
            tuple: Tuple containing arrays of states, actions, rewards, next states, and next rewards.
        """
        batch = random.sample(self.replay_buffer, batch_size)
        states = np.array([b[0] for b in batch]).reshape(batch_size, self.state_size)
        actions = np.array([b[1] for b in batch])
        rewards = [b[2] for b in batch]
        next_states = np.array([b[3] for b in batch]).reshape(batch_size, self.state_size)
        next_rewards = [b[4] for b in batch]
        return (states, actions, rewards, next_states, next_rewards)

    def act(self, state, time):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (State): Current state.
            time (int): Current time step.

        Returns:
            Action: Selected action.
        """
        epsilon = 0.1 ** (4*time / 100_000)
        random_action = self.int_to_action[np.random.randint(self.action_size)]
        state = np.array([state.q, state.theta]).reshape((1, self.state_size))
        Q_values = self.local_network.predict(state, verbose=0)
        greedy_action = self.int_to_action[np.argmax(Q_values[0])]
        return np.random.choice([random_action, greedy_action], p = [epsilon, 1 - epsilon])

    def train(self, batch_size):
        """
        Trains the local network using a batch of experiences.

        Args:
            batch_size (int): Size of the batch to train on.
        """
        state_batch, action_batch, reward_batch, next_state_batch, next_reward_batch = self.get_batch(batch_size)
        next_Q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_Q_values = np.amax(next_Q_values, axis=1)
        Q_values = self.local_network.predict(state_batch, verbose=0)
        for i in range(batch_size):
            action = self.action_to_int[Action(action_batch[i][0], action_batch[i][1])]
            Q_values[i][action] = reward_batch[i] + self.gamma * next_reward_batch[i] + (self.gamma**2) * max_next_Q_values[i]
        self.local_network.fit(state_batch, Q_values, verbose=0)