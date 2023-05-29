import matplotlib.pyplot as plt
import numpy as np

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


def visualize_convergence_over_T(pi1_t, pi2_t):
    group_size = 5
    pi1_t_grouped = pi1_t.reshape(-1, group_size)
    pi2_t_grouped = pi2_t.reshape(-1, group_size)

    # Calculate mean and variance for each group
    pi1_t_avg = np.mean(pi1_t_grouped, axis=1)
    pi1_t_var = np.var(pi1_t_grouped, axis=1)
    pi2_t_avg = np.mean(pi2_t_grouped, axis=1)
    pi2_t_var = np.var(pi2_t_grouped, axis=1)

    # Create x-axis values
    x = np.arange(0, 500001, 50000)

    # Plot mean values
    plt.plot(x, pi1_t_avg, label='pi1_t')
    plt.plot(x, pi2_t_avg, label='pi2_t')

    # Add error bars for variance
    plt.errorbar(x, pi1_t_avg, yerr=np.sqrt(pi1_t_var), linestyle='None', marker='o', label='Variance pi1_t')
    plt.errorbar(x, pi2_t_avg, yerr=np.sqrt(pi2_t_var), linestyle='None', marker='o', label='Variance pi2_t')

    # Add labels and title
    plt.xlabel('X Values')
    plt.ylabel('Averages')
    plt.title('Averages and Variance for Grouped Data')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()
