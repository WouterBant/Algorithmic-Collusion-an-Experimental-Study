from collections import defaultdict
import numpy as np


class Agent:
    """ Q-learning agent in this Multi-Agent Reinforcement Learning setting. """

    def __init__(self, actions, gamma):
        """
        Initializes an Agent.

        Args:
            actions (np array): List of available actions.
            gamma (float): Discount factor for Q-learning.
        """
        self.actions = actions
        self.Q = defaultdict(lambda: dict(zip(self.actions, np.zeros(len(self.actions)))))
        self.gamma = gamma

    def learn(self, state, action, next_state, profit, next_profit, time):
        """
        Updates the Q-function of the agent based on the observed transition.

        Args:
            state (Action object): Current state of the environment.
            action (Action object): Action taken by the agent in the current state.
            next_state (Action object): Next state of the environment after taking the action.
            profit (float): Current profit obtained from the transition.
            next_profit (float): Profit obtained from the next state.
            time (int): Current time step.
        """
        alpha = 0.6 - 0.5 * time / 500_000
        v = max(self.Q[next_state].values())
        self.Q[state][action] += alpha * (profit + self.gamma * next_profit
                                          + self.gamma**2 * v - self.Q[state][action])

    def act(self, state, time):
        """
        Determines the action that should be taken based on the epsilon-greedy policy.

        Args:
            state (Action object): Current state of the environment.
            time (int): Current time step.

        Returns:
            Action object: Action to be taken by the agent.
        """
        epsilon = 0.1 ** (4*time / 500_000)
        random_action = np.random.choice(self.actions)
        greedy_action = max(self.Q[state], key = self.Q[state].get)
        return np.random.choice([random_action, greedy_action], p = [epsilon, 1 - epsilon])
