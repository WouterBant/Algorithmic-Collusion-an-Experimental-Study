from collections import deque
from statistics import mean


class Regulator:
    """ Regulator agent in the artificial market. """

    def __init__(self, env, omega=1, kappa=3, delta=1.1, production_quota=0.15, CSR_quota=0.15, evaluation_period=5):
        """
        Initializes a Regulator.

        Args:
            actions (np array): List of available actions.
        """
        # Parameters in the penalty equation
        self.omega = omega
        self.kappa = kappa
        self.delta = delta

        # Thresholds for (acceptable) collusion
        self.production_quota = production_quota
        self.CSR_quota = CSR_quota

        # Collusive and competitive thresholds
        self.collusive_theta, self.competitive_theta = env.get_theta()
        self.collusive_q, self.competitive_q = env.get_q()

        # Used to keep track of actions taken
        self.consecutive_violations = -1
        self.thetas = deque(maxlen=evaluation_period)
        self.qs = deque(maxlen=evaluation_period)

        # Budget of the regulator
        self.budget = 0

        self.most_recent_reward = 0

    def enforce_regulations(self, action):
        """
        Enforces regulations on the firms in the market.

        Args:
            action (Action): The action containing theta and q values to enforce.
            
        Returns:
            float: The penalty value for violating regulations.
        """
        self.thetas.append(action.theta)
        self.qs.append(action.q)
        self.most_recent_reward = self.compute_reward()
        return self.most_recent_reward
        
    def compute_reward(self):
        """
        Computes the penalty for violating regulations based on the accumulated theta and q values.
        Penalties increase exponentially when quotas are consecutively not met.

        Returns:
            float: The penalty value.
        """
        q_stand = (mean(self.qs) - self.competitive_q) / (self.collusive_q - self.competitive_q)
        theta_stand = (mean(self.thetas) - self.competitive_theta) / (self.collusive_theta - self.competitive_theta)
        reward_h = self.omega * (self.production_quota - q_stand) + self.kappa * (self.CSR_quota - theta_stand)

        # Determine if there is a violation by the sign of reward_h
        if reward_h >= 0:
            self.consecutive_violations = -1
            reward = reward_h * (self.delta ** self.consecutive_violations)

            # Do not reward when there is no budget for it
            if reward > self.budget:
                reward = 0
            else:
                self.budget -= 2*reward
        else:
            self.consecutive_violations += 1
            reward = reward_h * (self.delta ** self.consecutive_violations)
            self.budget -= 2*reward  # Increase in budget when violation
        return reward
