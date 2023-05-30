from enum import Enum
import statistics


class Metric(Enum):
    Profit = 1
    Theta = 2
    Production_quantity = 3


class Cycle_Classifier:
    """ Class for cycle classification and analysis. """

    def __init__(self, pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L):
        """
        Initializes the Cycle_Classifier object.

        Args:
            pi1_L (list): List of floats.
            pi2_L (list): List of floats.
            theta1_L (list): List of floats.
            theta2_L (list): List of floats.
            q1_L (list): List of floats.
            q2_L (list): List of floats.
        """
        self.found_cycles = dict()
        self.create_dictionary(pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)

    def identify_cycle(self, cur_pi1, cur_pi2, cur_theta1, cur_theta2, cur_q1, cur_q2):
        """
        Identifies a cycle at the end of a given episode.

        Args:
            cur_pi1 (list): List of floats.
            cur_pi2 (list): List of floats.
            cur_theta1 (list): List of floats.
            cur_theta2 (list): List of floats.
            cur_q1 (list): List of floats.
            cur_q2 (list): List of floats.

        Returns:
            list: List of tuples representing the identified cycle.
        """
        start = len(cur_pi1)-1
        cycle_found = False
        while not cycle_found:
            cycle = []
            for t in range(start, -1, -1):
                cur_play = (cur_pi1[t], cur_pi2[t], cur_theta1[t], cur_theta2[t], cur_q1[t], cur_q2[t])
                # When the current state was also the first state, a cycle has been found
                if len(cycle) > 0 and cur_play == cycle[0]:
                    cycle_found = True
                    break
                # Otherwise add the current state to the cycle being created
                cycle.append(cur_play)
            start -= 15
        return cycle[::-1]
    
    def alternative_cycle(self, pattern):
        """
        Generates the cycle pattern, but now players are switched.

        Args:
            pattern (list): List of tuples representing a cycle.

        Returns:
            list: List of tuples representing the alternative cycle pattern.
        """
        cycle = []
        for a, b, c, d, e, f in pattern:
            cycle.append((b, a, d, c, f, e))
        return cycle

    def create_dictionary(self, pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L):
        """
        Creates a dictionary of cycles and their number of occurrences.

        Args:
            pi1_L (list): List of floats.
            pi2_L (list): List of floats.
            theta1_L (list): List of floats.
            theta2_L (list): List of floats.
            q1_L (list): List of floats.
            q2_L (list): List of floats.
        """
        for i in range(len(q2_L)):
            cycle = self.identify_cycle(pi1_L[i], pi2_L[i], theta1_L[i], theta2_L[i], q1_L[i], q2_L[i])
            cycle_alternative = self.alternative_cycle(cycle)
            cycle_already_seen = False

            # See if the current cycle - in any order - has already been seen
            for i in range(len(cycle)):
                possible_order = tuple(cycle[i:] + cycle[:i])
                if possible_order in self.found_cycles:
                    self.found_cycles[possible_order] += 1
                    cycle_already_seen = True
                    break

                possible_order_alternative = tuple(cycle_alternative[i:] + cycle_alternative[:i])
                if possible_order_alternative in self.found_cycles:
                    self.found_cycles[possible_order_alternative] += 1
                    cycle_already_seen = True
                    break

            if not cycle_already_seen:
                self.found_cycles[tuple(cycle)] = 1

    def number_of_cycles(self):
        """
        Identifies the number of different cycles found.
        
        Returns:
            int: Number of different cycles found in the 1,000 episodes.
        """
        return len(self.found_cycles)
    
    def most_found_cycle(self):
        """
        Identifies the most found cycle.

        Returns:
            tuple: Most found cycle and the number of times it is found.
        """
        most_found_cycle = max(self.found_cycles, key=self.found_cycles.get)
        number_of_times_found = self.found_cycles[most_found_cycle]
        return (most_found_cycle, number_of_times_found)
    
    def longest_cycles(self):
        """
        Identifies the longest cycles.

        Returns:
            list: List of tuples containing the longest cycles and their number of occurrences.
        """
        len_longest_cycles = 0
        longest_cycles = []
        for cycle in self.found_cycles.keys():
            if len(cycle) == len_longest_cycles:
                longest_cycles.append(cycle)
            elif len(cycle) > len_longest_cycles:
                len_longest_cycles = len(cycle)
                longest_cycles = [cycle]
        return [(cycle, self.found_cycles[cycle]) for cycle in longest_cycles]

    def most_profitable_cycle(self):
        """
        Identifies the most profitable cycles.

        Returns:
            list: Highest profit + list of tuples containing the highest profitable cycles and their number of occurrences.
        """
        highest_profit = 0
        highest_profitable_cycles = []
        for cycle in self.found_cycles.keys():
            cur_profit = 0
            for i in cycle:
                cur_profit += i[0] + i[1]
            cur_profit /= (2 * len(cycle))
            if cur_profit == highest_profit:
                highest_profitable_cycles.append(cycle)
            elif cur_profit > highest_profit:
                highest_profit = cur_profit
                highest_profitable_cycles = [cycle]
        return [highest_profit] + [(cycle, self.found_cycles[cycle]) for cycle in highest_profitable_cycles]
    
    def subcompetitive_profit_cycles(self):
        """
        Identifies all subcompetitive profit cycles.

        Returns:
            list: List of tuples containing cycles with subcompetitive profits and their number of occurrences.
        """
        subcompetitive_cycles = []
        for cycle in self.found_cycles.keys():
            cur_profit = 0
            for i in cycle:
                cur_profit += i[0] + i[1]
            cur_profit /= (2 * len(cycle))
            if cur_profit <= 3:
                subcompetitive_cycles.append(cycle)
        return [(cycle, self.found_cycles[cycle]) for cycle in subcompetitive_cycles]
    
    def mean_variance_calculator(self, metric):
        """
        Calculates the mean and variance of a given metric for all cycles, taking into account the number of occurrences.

        Args:
            metric (Metric): The metric to calculate mean and variance for.

        Returns:
            tuple: A tuple containing the mean and variance of the metric.
        """
        idx = 0 if metric == Metric.Profit else 2 if metric == Metric.Theta else 4
        vals = []
        for cycle in self.found_cycles.keys():
            som = 0
            for i in cycle:
                som += i[idx] + i[idx+1]
            v = som / (2 * len(cycle))
            for _ in range(self.found_cycles[cycle]):
                vals.append(v)
        return (statistics.mean(vals), statistics.variance(vals))
    
    def mean_variance_profit(self):
        """
        Calculates the mean and variance of the profit metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the profit metric.
        """
        return self.mean_variance_calculator(Metric.Profit)
    
    def mean_variance_theta(self):
        """
        Calculates the mean and variance of the theta metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the theta metric.
        """
        return self.mean_variance_calculator(Metric.Theta)
    
    def mean_variance_q(self):
        """
        Calculates the mean and variance of the production quantity metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the production quantity metric.
        """
        return self.mean_variance_calculator(Metric.Production_quantity)