class Action:
    def __init__(self, q, theta):
        """
        Initializes an action with a given quantity (q) and theta value.

        Args:
            q (float): Production quantity.
            theta (float): Theta value.
        """
        self.q = q
        self.theta = theta

    def __hash__(self):
        """
        Returns the hash value of the Action object based on its q and theta values.

        Returns:
            int: Hash value of the Action object.
        """
        return hash((self.q, self.theta))
