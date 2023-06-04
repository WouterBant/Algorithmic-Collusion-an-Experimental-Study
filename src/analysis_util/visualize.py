import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from analysis_util.cylcle_classifier import Cycle_Classifier


def visualize_last_L(pi1, pi2, theta1, theta2, q1, q2):
    plt.figure(figsize=(10, 6))

    # Plot for player 1
    plt.subplot(2, 1, 1)
    plt.plot(pi1, label='pi1'); plt.plot(theta1, label='theta1'); plt.plot(q1, label='q1')
    plt.legend()
    plt.title('Variables 1'); plt.xlabel('Index'); plt.ylabel('Value')

    # Plot for player 2
    plt.subplot(2, 1, 2)
    plt.plot(pi2, label='pi2'); plt.plot(theta2, label='theta2'); plt.plot(q2, label='q2')
    plt.legend()
    plt.title('Variables 2'); plt.xlabel('Index'); plt.ylabel('Value')

    # Display plots
    plt.tight_layout()
    plt.show()


def visualize_cycle(cycle):
    pi1_cycle, pi2_cycle, theta1_cycle, theta2_cycle, q1_cycle, q2_cycle = [], [], [], [], [], []
    for pi1, pi2, theta1, theta2, q1, q2 in cycle:
        pi1_cycle.append(pi1); pi2_cycle.append(pi2)
        theta1_cycle.append(theta1); theta2_cycle.append(theta2)
        q1_cycle.append(q1); q2_cycle.append(q2)
    a = 4 if len(pi1_cycle) == 1 else 2
    pi1_cycle *= a
    pi2_cycle *= a
    theta1_cycle *= a
    theta2_cycle *= a
    q1_cycle *= a
    q2_cycle *= a

    plt.figure(figsize=(10, 6))

    # Plot for player 1
    plt.subplot(2, 1, 1)
    plt.plot(pi1_cycle, label='pi1'); plt.plot(theta1_cycle, label='theta1'); plt.plot(q1_cycle, label='q1')
    plt.legend()
    plt.title('Variables 1'); plt.xlabel('Index'); plt.ylabel('Value')

    # Plot for player 2
    plt.subplot(2, 1, 2)
    plt.plot(pi2_cycle, label='pi2'); plt.plot(theta2_cycle, label='theta2'); plt.plot(q2_cycle, label='q2')
    plt.legend()
    plt.title('Variables 2'); plt.xlabel('Index'); plt.ylabel('Value')

    # Display plots
    plt.tight_layout()
    plt.show()


def visualize_convergence_over_T(env, groupname, gamma):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, '..', '..', 'data', 'simulation_data.h5')

    file = h5py.File(file_path, 'r')

    group = file[groupname]

    pi1_t = group['pi1_t'][:]; pi2_t = group['pi2_t'][:]
    theta1_t = group['theta1_t'][:]; theta2_t = group['theta2_t'][:]
    q1_t = group['q1_t'][:]; q2_t = group['q2_t'][:]

    file.close()

    # Example profit (1000 by 55 array)
    profit = np.vstack((pi1_t, pi2_t))
    collusive_profit, competitive_profit = env.get_profit()
    profit = (profit - competitive_profit) / (collusive_profit - competitive_profit)

    # Compute the mean and standard deviation along axis 0 for profit
    mean_profit = np.mean(profit, axis=0)
    std_profit = np.std(profit, axis=0)

    # Calculate the confidence interval for profit
    confidence_interval_profit = 1.96 * std_profit / np.sqrt(2000)

    # Example q (1000 by 55 array)
    q = np.vstack((q1_t, q2_t))
    collusive_q, competitive_q = env.get_q()
    q = (q - competitive_q) / (collusive_q - competitive_q)

    # Compute the mean and standard deviation along axis 0 for q
    mean_q = np.mean(q, axis=0)
    std_q = np.std(q, axis=0)

    # Calculate the confidence interval for q
    confidence_interval_q = 1.96 * std_q / np.sqrt(2000)

    # Example theta (1000 by 55 array)
    theta = np.vstack((theta1_t, theta2_t))
    collusive_theta, competitive_theta = env.get_theta()
    theta = (theta - competitive_theta) / (collusive_theta - competitive_theta)

    # Compute the mean and standard deviation along axis 0 for theta
    mean_theta = np.mean(theta, axis=0)
    std_theta = np.std(theta, axis=0)

    # Calculate the confidence interval for theta
    confidence_interval_theta = 1.96 * std_theta / np.sqrt(2000)

    # Plotting the mean with confidence intervals
    x = np.linspace(0, 500000, 55) / 1000  # x-axis values
    fig, ax = plt.subplots()

    # Plotting the mean values for profit
    ax.plot(x, mean_profit, color='b', label='∆')
    ax.fill_between(x, mean_profit - confidence_interval_profit, mean_profit + confidence_interval_profit,
                    color='b', alpha=0.1)

    # Plotting the mean values for q
    ax.plot(x, mean_q, color='g', label='Ψ')
    ax.fill_between(x, mean_q - confidence_interval_q, mean_q + confidence_interval_q,
                    color='g', alpha=0.1)

    # Plotting the mean values for theta
    ax.plot(x, mean_theta, color='r', label='Υ')
    ax.fill_between(x, mean_theta - confidence_interval_theta, mean_theta + confidence_interval_theta,
                    color='r', alpha=0.1)

    plt.xlabel('Number of Iterations (in 1000s)')
    plt.title('Convergence over Time')
    text = f'γ = {gamma}, ξ = {env.Xi}, μ = {env.Mu}, λ = {env.Lambda}, φ = {env.Phi}'
    plt.annotate(text, xy=(0.5, 0.05), xycoords='axes fraction', ha='center', va='bottom')
    plt.legend()
    plt.show()

def visualize_over_gamma(env, gamma_files):
    profit = []
    q = []
    theta = []

    for file_name in gamma_files:
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, '..', '..', 'data', 'simulation_data.h5')

        file = h5py.File(file_path, 'r')

        group = file[file_name]

        pi1_L = group['pi1_L'][:]; pi2_L = group['pi2_L'][:]
        theta1_L = group['theta1_L'][:]; theta2_L = group['theta2_L'][:]
        q1_L = group['q1_L'][:]; q2_L = group['q2_L'][:]

        file.close()
        Cycles = Cycle_Classifier(env, pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)
        profit.append(Cycles.mean_variance_profit())
        q.append(Cycles.mean_variance_q())
        theta.append(Cycles.mean_variance_theta())

    profit_mean = np.array([])
    profit_variance = np.array([])
    theta_mean = np.array([])
    theta_variance = np.array([])
    q_mean = np.array([])
    q_variance = np.array([])

    for m, var in profit:
        profit_mean = np.append(profit_mean, m)
        profit_variance = np.append(profit_variance, var)
    for m, var in theta:
        theta_mean = np.append(theta_mean, m)
        theta_variance = np.append(theta_variance, var)
    for m, var in q:
        q_mean = np.append(q_mean, m)
        q_variance = np.append(q_variance, var)

    x = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 0.98])  # x values
    profit_ci = 1.96 * np.sqrt(profit_variance) / np.sqrt(1000)  # profit confidence interval
    theta_ci = 1.96 * np.sqrt(theta_variance) / np.sqrt(1000)  # theta confidence interval
    q_ci = 1.96 * np.sqrt(q_variance) / np.sqrt(1000)  # q confidence interval

    fig, ax = plt.subplots()

    # Plotting profit
    ax.plot(x, profit_mean, color='b', label='Δ')
    ax.fill_between(x, (profit_mean - profit_ci), (profit_mean + profit_ci), color='b', alpha=.1)

    # Plotting q
    ax.plot(x, q_mean, color='g', label='Ψ')
    ax.fill_between(x, (q_mean - q_ci), (q_mean + q_ci), color='g', alpha=.1)

    # Plotting theta
    ax.plot(x, theta_mean, color='r', label='Υ')
    ax.fill_between(x, (theta_mean - theta_ci), (theta_mean + theta_ci), color='r', alpha=.1)

    # Adding legend and showing the plot
    ax.legend()
    plt.xlabel('Discount Factor (γ)')
    plt.title("Converged Metric Values for different Discount Factors (γ's)")
    text = f'ξ = {env.Xi}, μ = {env.Mu}, λ = {env.Lambda}, φ = {env.Phi}'
    plt.annotate(text, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom')
    plt.show()