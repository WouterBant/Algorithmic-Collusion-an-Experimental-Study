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

    def __eq__(self, other):
        """
        Compares two Action objects for equality based on their q and theta values.

        Args:
            other (Action): The other Action object to compare.

        Returns:
            bool: True if the Action objects are equal, False otherwise.
        """
        return isinstance(other, Action) and self.q == other.q and self.theta == other.theta
    
    def __repr__(self):
        """
        Returns a string representation of the Action object.

        Returns:
            str: String representation of the Action object.
        """
        return f"Action(q={self.q}, theta={self.theta})"