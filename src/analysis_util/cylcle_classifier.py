from enum import Enum
import statistics


class Metric(Enum):
    Profit = 1
    Theta = 2
    Production_quantity = 3


class Cycle_Classifier:
    """ Class for cycle classification and analysis. """

    def __init__(self, env, pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L):
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
        self.env = env
        self.collusive_profit, self.competitive_profit = env.get_profit()
        self.found_cycles = dict()
        self.total_number_of_cycles = len(pi1_L)
        self.taking_turns = 0
        # pi1_L, pi2_L, theta1_L, theta2_L = self.clean_data(pi1_L, pi2_L, theta1_L, theta2_L)  Does not influence results
        self.create_dictionary(pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)

    def clean_data(self, pi1_L, pi2_L, theta1_L, theta2_L):
        """
        Rounds the values in the input lists to three decimal places.

        Args:
            pi1_L (list): A list of lists representing the pi1 values.
            pi2_L (list): A list of lists representing the pi2 values.
            theta1_L (list): A list of lists representing the theta1 values.
            theta2_L (list): A list of lists representing the theta2 values.

        Returns:
            tuple: A tuple containing the modified lists pi1_L, pi2_L, theta1_L, and theta2_L.
        """
        for r in range(len(pi1_L)):
            for c in range(len(pi1_L[0])):
                pi1_L[r][c] = round(pi1_L[r][c], 3)
                pi2_L[r][c] = round(pi2_L[r][c], 3)
                theta1_L[r][c] = round(theta1_L[r][c], 3)
                theta2_L[r][c] = round(theta2_L[r][c], 3)
        return (pi1_L, pi2_L, theta1_L, theta2_L)

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
        q01 = False
        q02 = False
        while not cycle_found:
            cycle = []
            for t in range(start, -1, -1):
                cur_play = (cur_pi1[t], cur_pi2[t], cur_theta1[t], cur_theta2[t], cur_q1[t], cur_q2[t])
                if cur_q1[t] == 0:
                    q01 = True
                if cur_q2[t] == 0:
                    q02 = True
                # When the current state was also the first state, a cycle has been found
                if len(cycle) > 0 and cur_play == cycle[0]:
                    cycle_found = True
                    break
                # Otherwise add the current state to the cycle being created
                cycle.append(cur_play)
            start -= 15
        if q01 and q02:
            self.taking_turns += 1
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
        return len(self.found_cycles) / self.total_number_of_cycles
    
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
        return [highest_profit] + [(cycle, self.found_cycles[cycle]/self.total_number_of_cycles) for cycle in highest_profitable_cycles]
    
    def highest_possible_profits(self):
        """
        Calculates the highest possible profits based on the most profitable cycle.

        Returns:
            int: The highest possible profits if the most profitable cycle matches the collusive profit.
            0: If the most profitable cycle does not match the collusive profit.
        """
        highest = self.most_profitable_cycle()
        if highest[0] == self.collusive_profit:
            return highest[1][1]
        return 0
    
    def taking_turns_found(self):
        """
        Returns:
            Proportion of times where firms set q=0 in turn.
        """
        return self.taking_turns / self.total_number_of_cycles

    def subcompetitive_profit_cycles(self):
        """
        Identifies all subcompetitive profit cycles.

        Returns:
            list: List of tuples containing cycles with subcompetitive profits and their number of occurrences.
        """
        subcompetitive_cycles = []
        res = 0
        for cycle in self.found_cycles.keys():
            cur_profit = 0
            for i in cycle:
                cur_profit += i[0] + i[1]
            cur_profit /= (2 * len(cycle))
            if cur_profit <= self.competitive_profit:
                subcompetitive_cycles.append(cycle)
                res += self.found_cycles[cycle]
        return [res / self.total_number_of_cycles] + [(cycle, self.found_cycles[cycle]) for cycle in subcompetitive_cycles]
    
    def unilateral_subcompetitive_profit_cycles(self):
        """
        Returns:
            float: The proportion of times minimum of the individual average profit gain is below the competitive level.
        """
        res = 0
        for cycle in self.found_cycles.keys():
            cur_profit1 = 0
            cur_profit2 = 0
            for i in cycle:
                cur_profit1 += i[0]
                cur_profit2 += i[1]
            cur_profit1 /= len(cycle)
            cur_profit2 /= len(cycle)
            if cur_profit1 <= self.competitive_profit or cur_profit2 <= self.competitive_profit:
                res += self.found_cycles[cycle]
        return res / self.total_number_of_cycles
    
    def standardized_mean_variance_calculator(self, metric):
        """
        Calculates the mean and variance of a given metric for all cycles, taking into account the number of occurrences.

        Args:
            metric (Metric): The metric to calculate mean and variance for.

        Returns:
            tuple: A tuple containing the mean and variance of the metric.
        """
        if metric == Metric.Profit:
            collusive_value, competitive_value = self.env.get_profit()
            idx = 0
        elif metric == Metric.Theta:
            collusive_value, competitive_value = self.env.get_theta()
            idx = 2
        else:
            collusive_value, competitive_value = self.env.get_q()
            idx = 4
        
        vals = []
        for cycle in self.found_cycles.keys():
            som = 0
            for i in cycle:
                som += i[idx] + i[idx+1]
            a = som / (2 * len(cycle))
            v = (a - competitive_value) / (collusive_value - competitive_value)
            for _ in range(self.found_cycles[cycle]):
                vals.append(v)
        return (statistics.mean(vals), statistics.variance(vals))
    
    def cycles_length_one(self):
        """
        Returns:
            float: The proportion of times the actions cycle consisted of one action of both players.
        """
        res = 0
        for cycle in self.found_cycles.keys():
            if len(cycle) == 1:
                res += 1
        return res / self.total_number_of_cycles
    
    def mean_variance_profit(self):
        """
        Calculates the mean and variance of the profit metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the profit metric.
        """
        return self.standardized_mean_variance_calculator(Metric.Profit)
    
    def mean_variance_theta(self):
        """
        Calculates the mean and variance of the theta metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the theta metric.
        """
        return self.standardized_mean_variance_calculator(Metric.Theta)
    
    def mean_variance_q(self):
        """
        Calculates the mean and variance of the production quantity metric for all cycles, taking into account the number of occurrences.

        Returns:
            tuple: A tuple containing the mean and variance of the production quantity metric.
        """
        return self.standardized_mean_variance_calculator(Metric.Production_quantity)