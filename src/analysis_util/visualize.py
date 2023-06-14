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

    profit = np.array([])
    q = np.array([])
    theta = np.array([])

    profit_h = np.vstack((pi1_t, pi2_t))
    q_h = np.vstack((q1_t, q2_t))
    theta_h = np.vstack((theta1_t, theta2_t))

    for r in range(len(profit_h)):
        row_profit = np.array([])
        row_q = np.array([])
        row_theta = np.array([])
        
        for c in range(11):
            avg_profit = np.mean(profit_h[r, c*5:c*5+5])
            avg_q = np.mean(q_h[r, c*5:c*5+5])
            avg_theta = np.mean(theta_h[r, c*5:c*5+5])
            row_profit = np.append(row_profit, avg_profit)
            row_q = np.append(row_q, avg_q)
            row_theta = np.append(row_theta, avg_theta)
        
        profit = np.vstack((profit, row_profit)) if profit.size else row_profit
        q = np.vstack((q, row_q)) if q.size else row_q
        theta = np.vstack((theta, row_theta)) if theta.size else row_theta
                
    collusive_profit, competitive_profit = env.get_profit()
    profit = (profit - competitive_profit) / (collusive_profit - competitive_profit)

    mean_profit = np.mean(profit, axis=0)
    std_profit = np.std(profit, axis=0)

    l = len(profit)
    
    confidence_interval_profit = 1.96 * std_profit / np.sqrt(l)

    collusive_q, competitive_q = env.get_q()
    q = (q - competitive_q) / (collusive_q - competitive_q)

    mean_q = np.mean(q, axis=0)
    std_q = np.std(q, axis=0)

    confidence_interval_q = 1.96 * std_q / np.sqrt(l)
    
    collusive_theta, competitive_theta = env.get_theta()
    theta = (theta - competitive_theta) / (collusive_theta - competitive_theta)

    mean_theta = np.mean(theta, axis=0)
    std_theta = np.std(theta, axis=0)

    confidence_interval_theta = 1.96 * std_theta / np.sqrt(l)

    x = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    fig, ax = plt.subplots()

    ax.plot(x, mean_profit.T, 'bo--', label='∆')
    ax.fill_between(x, mean_profit.T - confidence_interval_profit.T, mean_profit.T + confidence_interval_profit.T,
                    color='b', alpha=0.1)

    ax.plot(x, mean_q.T, 'go--', label='Ψ')
    ax.fill_between(x, mean_q.T - confidence_interval_q.T, mean_q.T + confidence_interval_q.T,
                    color='g', alpha=0.1)

    ax.plot(x, mean_theta.T, 'ro--', label='Υ')
    ax.fill_between(x, mean_theta.T - confidence_interval_theta.T, mean_theta.T + confidence_interval_theta.T,
                    color='r', alpha=0.1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Number of Iterations (in 1000s)', fontsize=12)
    plt.title('Convergence During Episodes', fontsize=15)
    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=13)
    text = f'γ = {gamma}, ξ = {env.Xi}, μ = {env.Mu}, λ = {env.Lambda}, φ = {env.Phi}'
    plt.annotate(text, xy=(0.5, 0.05), xycoords='axes fraction', ha='center', va='bottom', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

def visualize_over_gamma(env, gamma_files, extra_space=0):
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

    x = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 0.98]) 
    profit_ci = 1.96 * np.sqrt(profit_variance) / np.sqrt(1000)
    theta_ci = 1.96 * np.sqrt(theta_variance) / np.sqrt(1000)
    q_ci = 1.96 * np.sqrt(q_variance) / np.sqrt(1000)
    fig, ax = plt.subplots()

    ax.plot(x, profit_mean, 'bo--', label='Δ')
    ax.plot(x, q_mean, 'go--', label='Ψ')
    ax.plot(x, theta_mean, 'ro--', label='Υ')

    ax.fill_between(x, (profit_mean - profit_ci), (profit_mean + profit_ci), color='b', alpha=0.1)
    ax.fill_between(x, (q_mean - q_ci), (q_mean + q_ci), color='g', alpha=0.1)
    ax.fill_between(x, (theta_mean - theta_ci), (theta_mean + theta_ci), color='r', alpha=0.1)

    # Adding legend and showing the plot
    ax.legend(loc='lower right', fontsize=12)
    ax.axvline(x=0.9, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Discount Factor (γ)', fontsize=12)
    plt.title("Converged Metric Values", fontsize=15)
    plt.xlim(0.74, 0.99)
    plt.xticks([0.75, 0.8, 0.85, 0.9, 0.95, 0.98], fontsize=12)
    plt.yticks(fontsize=13)
    text = f'ξ = {env.Xi}, μ = {env.Mu}, λ = {env.Lambda}, φ = {env.Phi}'
    plt.annotate(text, xy=(0.50+extra_space, 0.90), xycoords='axes fraction', ha='right', va='bottom', fontsize=13)
    plt.show()