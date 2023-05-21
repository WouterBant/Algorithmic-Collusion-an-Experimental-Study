from collections import deque


class Regulator:
    """ Regulator agent in the artificial market. """

    def __init__(self, production_quota=1.8, CSR_quota=0.4, evaluation_period=5):
        """
        Initializes a Regulator.

        Args:
            actions (np array): List of available actions.
        """
        self.evaluation_period = evaluation_period
        self.production_quota = production_quota
        self.CSR_quota = CSR_quota
        self.consecutive_violations = 0
        self.theta_sum = self.q_sum = 0
        self.thetas = deque()
        self.qs = deque()

    def enforce_regulations(self, action):
        """
        Enforces regulations on the firms in the market.

        Args:
            action (Action): The action containing theta and q values to enforce.
            
        Returns:
            float: The penalty value for violating regulations.
        """
        if len(self.thetas) < 5:
            self.thetas.append(action.theta)
            self.qs.append(action.q)
            return 0
        out_theta, out_q = self.thetas.popleft(), self.qs.popleft()
        self.theta_sum += action.theta - out_theta
        self.q_sum += action.q - out_q
        penalty = self.compute_penalty()
        if penalty == 0:
            self.consecutive_violations = 0
        else:
            self.consecutive_violations += 1
        return - penalty
        
    def compute_penalty(self):
        """
        Computes the penalty for violating regulations based on the accumulated theta and q values.
        Penalties increase exponentially when quotas are consecutively not met.

        Returns:
            float: The penalty value.
        """
        penalty_theta = max(self.theta_sum/self.evaluation_period - self.CSR_quota, 0)
        penalty_q = max(self.production_quota - self.theta_sum/self.evaluation_period, 0)
        multiplier = 1.1**(self.consecutive_violations+1)
        penalty = (penalty_theta + penalty_q) * multiplier
        return penalty
