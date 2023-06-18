from classes.environment import Environment
from analysis_util.cylcle_classifier import Cycle_Classifier
import os
import h5py
import csv

def get_statistics(filenames, Lambda=1, DQN=False):
    current_dir = os.getcwd()
    file_path1 = os.path.join(current_dir, '..', '..', 'data', 'statistics.csv')
    file_path2 = os.path.join(current_dir, '..', '..', 'data', 'simulation_data.h5')
    with open(file_path1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Mean Delta', 'Variance Delta', 'Mean Psi', 'Variance Psi', 'Mean Upsilon', 'Variance Upsilon',
                         'Unique Cycles', 'Subcompetitive Profit Cycles',
                         'Unilateral Subcompetitive Profit Cycles', 'Highest Possible Profits', 'Number of times taking turns in production'])
        
        for file_name in filenames:
            file = h5py.File(file_path2, 'r')
            group = file[file_name]
            if DQN:
                pi1_L = group['pi1_t'][:]
                pi2_L = group['pi2_t'][:]
                theta1_L = group['theta1_t'][:]
                theta2_L = group['theta2_t'][:]
                q1_L = group['q1_t'][:]
                q2_L = group['q2_t'][:]
            else:
                pi1_L = group['pi1_L'][:]
                pi2_L = group['pi2_L'][:]
                theta1_L = group['theta1_L'][:]
                theta2_L = group['theta2_L'][:]
                q1_L = group['q1_L'][:]
                q2_L = group['q2_L'][:]

            file.close()

            env = Environment(Lambda=Lambda)
            Cycles = Cycle_Classifier(env, pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)

            profit = Cycles.mean_variance_profit()
            q = Cycles.mean_variance_q()
            theta = Cycles.mean_variance_theta()

            # In proportions of total number of episodes
            subcompetitive_profit_cycles = Cycles.subcompetitive_profit_cycles()[0]
            num_cycles = Cycles.number_of_cycles()
            unilateral_subcompetitive_profit_cycles = Cycles.unilateral_subcompetitive_profit_cycles()
            highest_possible_profits = Cycles.highest_possible_profits()
            num_taking_turns = Cycles.taking_turns_found()

            # Round the numbers to 3 decimal places
            rounded_profit = [round(value, 3) for value in profit]
            rounded_q = [round(value, 3) for value in q]
            rounded_theta = [round(value, 3) for value in theta]
            rounded_subcompetitive_profit_cycles = round(subcompetitive_profit_cycles, 3)
            rounded_num_cycles = round(num_cycles, 3)
            rounded_unilateral_subcompetitive_profit_cycles = round(unilateral_subcompetitive_profit_cycles, 3)
            rounded_highest_possible_profits = round(highest_possible_profits, 3)

            # Write the statistics to the CSV file
            writer.writerow([file_name] + rounded_profit + rounded_q + rounded_theta +
                             [rounded_num_cycles, rounded_subcompetitive_profit_cycles,
                              rounded_unilateral_subcompetitive_profit_cycles, rounded_highest_possible_profits, num_taking_turns])

    print("Statistics have been written to statistics.csv file.")