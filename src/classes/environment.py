class Environment:
    """ Economic environment. """

    def __init__(self, Xi=6, Mu=1, Lambda=1, Phi=1):
        """
        Initializes the economic environment.

        Args:
            Xi (float): Constant value for the price calculation (default: 6).
            Mu (float): Constant value for the price calculation (default: 1).
            Lambda (float): Constant value for the price calculation (default: 1).
            Phi (float): Constant value for the cost calculation (default: 1).
        """
        self.Xi = Xi
        self.Mu = Mu
        self.Lambda = Lambda
        self.Phi = Phi
 
    def P(self, q1, q2, Theta1, Theta2):
        """
        Calculates and returns the price for the goods of firm 1 and 2.

        Args:
            q1 (float): Production quantity of firm 1.
            q2 (float): Production quantity of firm 2.
            Theta1 (float): Theta value of firm 1.
            Theta2 (float): Theta value of firm 2.

        Returns:
            tuple: A tuple containing the prices for the goods of firm 1 and 2.
        """
        p1 = max(self.Xi - self.Mu * (q1 + q2) + self.Lambda * (Theta2 - Theta1), 0)
        p2 = max(self.Xi - self.Mu * (q1 + q2) + self.Lambda * (Theta1 - Theta2), 0)
        return (p1, p2)

    def C(self, Theta1, Theta2):
        """
        Calculates and returns the costs for firm 1 and 2.

        Args:
            Theta1 (float): Theta value of firm 1.
            Theta2 (float): Theta value of firm 2.

        Returns:
            tuple: A tuple containing the costs for firm 1 and 2.
        """
        c1 = self.Phi * (1-Theta1)**2
        c2 = self.Phi * (1-Theta2)**2
        return (c1, c2)

    def Pi(self, action1, action2):
        """
        Calculates and returns the profits for firm 1 and 2.

        Args:
            action1 (Action object): Action taken by firm 1.
            action2 (Action object): Action taken by firm 2.

        Returns:
            tuple: A tuple containing the profits for firm 1 and 2.
        """
        p1, p2 = self.P(action1.q, action2.q, action1.theta, action2.theta)
        c1, c2 = self.C(action1.theta, action2.theta)
        pi1, pi2 = action1.q*p1-c1, action2.q*p2-c2
        return (pi1, pi2)
    
    def get_profit(self):
        """
        Calculates the profit values for the collusive and competitive cases.

        Returns:
            tuple: A tuple containing the profit value for the collusive case and the competitive case.
        """
        collusive_value = (self.Xi**2) / (8 * self.Mu)
        competitive_value = ((self.Xi**2) / (9 * self.Mu)) * (1 - (self.Lambda**2) / (4 * self.Mu * self.Phi))
        return (collusive_value, competitive_value)
    
    def get_theta(self):
        """
        Calculates the theta values for the collusive and competitive cases.

        Returns:
            tuple: A tuple containing the theta value for the collusive case and the competitive case.
        """
        collusive_value = 1
        competitive_value = 1 - (self.Lambda * self.Xi) / (6 * self.Mu * self.Phi)
        return (collusive_value, competitive_value)

    def get_q(self):
        """
        Calculates the q values for the collusive and competitive cases.

        Returns:
            tuple: A tuple containing the q value for the collusive case and the competitive case.
        """
        collusive_value = self.Xi / (4 * self.Mu)
        competitive_value = self.Xi / (3 * self.Mu)
        return (collusive_value, competitive_value)
