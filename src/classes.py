import numpy as np
from collections import defaultdict


class Agent:
    """ The decision maker (firm) in this Multi-Agent Reinforcement Learning setting. """

    def __init__(self, actions, gamma):
        self.actions = actions
        self.Q = defaultdict(lambda: dict(zip(self.actions, np.zeros(len(self.actions)))))
        self.gamma = gamma

    def learn(self, state, action, next_state, profit, next_profit, time):
        """ Updates the Q-function of the agent. """
        alpha = 0.6 - 0.5 * time / 500_000
        v = max(self.Q[next_state].values())
        self.Q[state][action] += alpha * (profit + self.gamma * next_profit
                                          + self.gamma**2 * v - self.Q[state][action])

    def act(self, state, time):
        """ Returns the action that should be taken according to the epsilon-greedy policy. """
        epsilon = 0.1 ** (4*time / 500_000)
        random_action = np.random.choice(self.actions)
        greedy_action = max(self.Q[state], key = self.Q[state].get)
        return np.random.choice([random_action, greedy_action], p = [epsilon, 1 - epsilon])
    

class Environment:
    """ Economic environment. """

    def __init__(self, Xi=6, Mu=1, Lambda=1, Phi=1):
        self.Xi = Xi
        self.Mu = Mu
        self.Lambda = Lambda
        self.Phi = Phi
 
    def P(self, q1, q2, Theta1, Theta2):
        """ Return the price for the good of firm 1 and 2 in a tuple. """
        p1 = self.Xi - self.Mu * (q1 + q2) + self.Lambda * (Theta2 - Theta1)
        p2 = self.Xi - self.Mu * (q1 + q2) + self.Lambda * (Theta1 - Theta2)
        return (p1, p2)

    def C(self, Theta1, Theta2):
        """ Return the costs for firm 1 and 2 in a tuple. """
        c1 = self.Phi * (1-Theta1)**2
        c2 = self.Phi * (1-Theta2)**2
        return (c1, c2)

    def Pi(self, action1, action2):
        """ Return the profit for firm 1 and 2 in a tuple. """
        p1, p2 = self.P(action1.q, action2.q, action1.theta, action2.theta)
        c1, c2 = self.C(action1.theta, action2.theta)
        pi1, pi2 = action1.q*p1-c1, action2.q*p2-c2
        return (pi1, pi2)
    

class Action:
    def __init__(self, q, theta):
        self.q = q
        self.theta = theta

    def __hash__(self):
        return hash((self.q, self.theta))