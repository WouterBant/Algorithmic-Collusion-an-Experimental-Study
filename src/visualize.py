import matplotlib.pyplot as plt


def visualize_last_L(pi1, pi2, theta1, theta2, q1, q2):
    # Variables 1 plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pi1, label='pi1')
    plt.plot(theta1, label='theta1')
    plt.plot(q1, label='q1')
    plt.legend()
    plt.title('Variables 1')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Variables 2 plot
    plt.subplot(2, 1, 2)
    plt.plot(pi2, label='pi2')
    plt.plot(theta2, label='theta2')
    plt.plot(q2, label='q2')
    plt.legend()
    plt.title('Variables 2')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()