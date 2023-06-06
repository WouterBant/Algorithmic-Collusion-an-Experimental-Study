from classes.action import Action
from classes.regulator import Regulator
from classes import Qlearning, DQN
import numpy as np
import os
import h5py


def create_actions(Qs, Thetas):
    """
    Creates a np array of possible actions based on:
        - The set of possible production quantities (Qs). 
        - The set of possible theta's (obtained from k).

    Args:
        Qs (np array): List of quantity values.
        k (int): Number of theta values.

    Returns:
        np array: An array containing the created actions as Action objects.
    """
    actions_h = [Action(q, theta) for q in Qs for theta in Thetas]
    actions_h.append(Action(0, 1))
    actions = np.array(actions_h)
    return actions


def simulate_episode_Qlearning(env, Qs, Thetas, gamma=0.9, T=500_000, L=100):
    """
    Simulates an episode of the game with Q-learning.

    Args:
        env (Environment): The economic environment.
        Qs (list): List of allowed production quantities.
        k (int): Parameter used to create the set of available actions (default: 6).
        gamma (float): Discount factor (default: 0.9).
        T (int): Number of iterations in an episode (default: 500,000).
        L (int): Number of last runs to collect metrics from (default: 1,000).

    Returns:
        tuple: A tuple containing the metrics from the last L runs:
            - pi1_L (list): List of profits for player 1.
            - pi2_L (list): List of profits for player 2.
            - theta1_L (list): List of theta values for player 1.
            - theta2_L (list): List of theta values for player 2.
            - q1_L (list): List of production quantities for player 1.
            - q2_L (list): List of production quantities for player 2.
    """
    # Create the set of available actions
    actions = create_actions(Qs, Thetas)

    # Pick the first two prices of both players randomly
    Action1, Action2, Action1_next, Action2_next = np.random.choice(actions, size=4)
    
    # Create both agents
    Agent1 = Qlearning.Agent(actions, gamma)
    Agent2 = Qlearning.Agent(actions, gamma)

    # Keep track of metrics
    pi1_ep, pi2_ep = [], []
    q1, q2 = [], []
    theta1, theta2 = [], []

    for t in range(3, T+1):
        if t % 2:
            # Get action and state
            current_action1 = Action1_next
            current_action2 = Action2

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)
            
            # With regulator subtract penalties here

            # Calculate the next profit
            pi1_next, _ = env.Pi(current_action1, Action2_next)

            # Update Q-function
            Agent1.learn(state=current_action2, action=current_action1, 
                         next_state=Action2_next, profit=pi1, 
                         next_profit=pi1_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action1, Action1_next = Action1_next, Agent1.act(state=Action2_next, time=t)
        else:
            # Get state and action
            current_action1 = Action1
            current_action2 = Action2_next

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)

            # Calculate the next profit
            _, pi2_next = env.Pi(Action1_next, current_action2)

            # Update Q-function
            Agent2.learn(state=current_action1, action=current_action2,
                         next_state=Action1_next, profit=pi2,
                         next_profit=pi2_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action2, Action2_next = Action2_next, Agent2.act(state=Action1_next, time=t)
        
        # Append the profits, q's and theta's
        pi1_ep.append(pi1); pi2_ep.append(pi2)
        q1.append(current_action1.q); theta1.append(current_action1.theta)
        q2.append(current_action2.q); theta2.append(current_action2.theta)
    
    # Get the profits, theta's, and q's from the last L runs
    pi1_L, pi2_L = pi1_ep[-L:], pi2_ep[-L:]
    theta1_L, theta2_L = theta1[-L:], theta2[-L:]
    q1_L, q2_L = q1[-L:], q2[-L:]

    pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = [], [], [], [], [], []
    for i in range(0, T, 49999):
        pi1_t += pi1_ep[i:i+5]
        pi2_t += pi2_ep[i:i+5]
        theta1_t += theta1[i:i+5]
        theta2_t += theta2[i:i+5]
        q1_t += q1[i:i+5]
        q2_t += q2[i:i+5]
    
    return (pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L,
            pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t, Agent1.Q, Agent2.Q)


def simulate_episode_DQN(env, Qs, Thetas, gamma=0.9, T=100_000, L=100):
    """
    Simulates an episode of the game with DQNs.

    Args:
        env (Environment): The economic environment.
        Qs (list): List of allowed production quantities.
        k (int): Parameter used to create the set of available actions (default: 6).
        gamma (float): Discount factor (default: 0.9).
        T (int): Number of iterations in an episode (default: 500,000).
        L (int): Number of last runs to collect metrics from (default: 1,000).

    Returns:
        tuple: A tuple containing the metrics from the last L runs:
            - pi1_L (list): List of profits for player 1.
            - pi2_L (list): List of profits for player 2.
            - theta1_L (list): List of theta values for player 1.
            - theta2_L (list): List of theta values for player 2.
            - q1_L (list): List of production quantities for player 1.
            - q2_L (list): List of production quantities for player 2.
    """
    # Create the set of available actions
    actions = create_actions(Qs, Thetas)

    # Pick the first two prices of both players randomly
    Action1, Action2, Action1_next, Action2_next = np.random.choice(actions, size=4)
    
    # Create both agents
    Agent1 = DQN.Agent(actions, gamma=gamma)
    Agent2 = DQN.Agent(actions, gamma=gamma)

    batch_size = 32

    # Keep track of metrics
    pi1_ep, pi2_ep = [], []
    q1, q2 = [], []
    theta1, theta2 = [], []

    for t in range(3, T+1):
        print(t)
        if t % 2:
            if (t-1) % Agent1.update_rate == 0:
                Agent1.update_target_network()
                
            # Get action and state
            current_action1 = Action1_next
            current_action2 = Action2

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)

            # Calculate the next profit
            pi1_next, _ = env.Pi(current_action1, Action2_next)

            Agent1.save_experience(state=current_action2, action=current_action1, reward=pi1, next_state=Action2_next, next_reward=pi1_next) # Save experience in ReplayBuffer

            # Update current action player 1, and determine next action based on the next state
            Action1, Action1_next = Action1_next, Agent1.act(state=Action2_next, time=t)
            
            if (t-1) % 500 == 0:
                Agent2.train(batch_size)
        else:
            if t % Agent2.update_rate == 0:
                Agent2.update_target_network()
                
            # Get state and action
            current_action1 = Action1
            current_action2 = Action2_next

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)

            # Calculate the next profit
            _, pi2_next = env.Pi(Action1_next, current_action2)

            Agent2.save_experience(state=current_action1, action=current_action2, reward=pi2, next_state=Action1_next, next_reward=pi2_next) # Save experience in ReplayBuffer

            # Update current action player 1, and determine next action based on the next state
            Action2, Action2_next = Action2_next, Agent2.act(state=Action1_next, time=t)
            
            if t % 500 == 0:
                Agent2.train(batch_size)
        
        # Append the profits, q's and theta's
        pi1_ep.append(pi1); pi2_ep.append(pi2)
        q1.append(current_action1.q); theta1.append(current_action1.theta)
        q2.append(current_action2.q); theta2.append(current_action2.theta)
    
    # Get the profits, theta's, and q's from the last L runs
    pi1_L, pi2_L = pi1_ep[-L:], pi2_ep[-L:]
    theta1_L, theta2_L = theta1[-L:], theta2[-L:]
    q1_L, q2_L = q1[-L:], q2[-L:]

    pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = [], [], [], [], [], []
    for i in range(0, T, 9999):
        pi1_t += pi1_ep[i:i+5]
        pi2_t += pi2_ep[i:i+5]
        theta1_t += theta1[i:i+5]
        theta2_t += theta2[i:i+5]
        q1_t += q1[i:i+5]
        q2_t += q2[i:i+5]
    
    return (pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L,
            pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t)


def simulate_episode_with_regulator(env, Qs, Thetas, gamma=0.9, T=500_000, L=100):
    """
    Simulates an episode of the game with a regulator.

    Args:
        env (Environment): The economic environment.
        Qs (list): List of allowed production quantities.
        k (int): Parameter used to create the set of available actions (default: 6).
        gamma (float): Discount factor (default: 0.9).
        T (int): Number of iterations in an episode (default: 500,000).
        L (int): Number of last runs to collect metrics from (default: 1,000).

    Returns:
        tuple: A tuple containing the metrics from the last L runs:
            - pi1_L (list): List of profits for player 1.
            - pi2_L (list): List of profits for player 2.
            - theta1_L (list): List of theta values for player 1.
            - theta2_L (list): List of theta values for player 2.
            - q1_L (list): List of production quantities for player 1.
            - q2_L (list): List of production quantities for player 2.
    """
    # Create the set of available actions
    actions = create_actions(Qs, Thetas)

    # Pick the first two prices of both players randomly
    Action1, Action2, Action1_next, Action2_next = np.random.choice(actions, size=4)
    
    # Create both agents
    Agent1 = Qlearning.Agent(actions, gamma)
    Agent2 = Qlearning.Agent(actions, gamma)

    # Create a regulator for both agents
    Regulator1 = Regulator(env)
    Regulator2 = Regulator(env)

    # Keep track of metrics
    pi1_ep, pi2_ep = [], []
    q1, q2 = [], []
    theta1, theta2 = [], []

    for t in range(3, T+1):
        if t % 2:
            # Get action and state
            current_action1 = Action1_next
            current_action2 = Action2

            # Compute the reward
            reward = Regulator1.enforce_regulations(current_action1)

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)
            
            pi1 += reward
            pi2 += Regulator2.most_recent_reward

            # Calculate the next profit
            pi1_next = env.Pi(current_action1, Action2_next)[0] + reward

            # Update Q-function
            Agent1.learn(state=current_action2, action=current_action1, 
                         next_state=Action2_next, profit=pi1, 
                         next_profit=pi1_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action1, Action1_next = Action1_next, Agent1.act(state=Action2_next, time=t)
        else:
            # Get state and action
            current_action1 = Action1
            current_action2 = Action2_next

            # Compute the reward
            reward = Regulator2.enforce_regulations(current_action2)

            # Calculate the current profits
            pi1, pi2 = env.Pi(current_action1, current_action2)

            pi1 += Regulator1.most_recent_reward
            pi2 += reward

            # Calculate the next profit
            pi2_next = env.Pi(Action1_next, current_action2)[1] + reward

            # Update Q-function
            Agent2.learn(state=current_action1, action=current_action2,
                         next_state=Action1_next, profit=pi2,
                         next_profit=pi2_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action2, Action2_next = Action2_next, Agent2.act(state=Action1_next, time=t)
        
        # Append the profits, q's and theta's
        pi1_ep.append(pi1); pi2_ep.append(pi2)
        q1.append(current_action1.q); theta1.append(current_action1.theta)
        q2.append(current_action2.q); theta2.append(current_action2.theta)
    
    # Get the profits, theta's, and q's from the last L runs
    pi1_L, pi2_L = pi1_ep[-L:], pi2_ep[-L:]
    theta1_L, theta2_L = theta1[-L:], theta2[-L:]
    q1_L, q2_L = q1[-L:], q2[-L:]

    pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = [], [], [], [], [], []
    for i in range(0, T, 49999):
        pi1_t += pi1_ep[i:i+5]
        pi2_t += pi2_ep[i:i+5]
        theta1_t += theta1[i:i+5]
        theta2_t += theta2[i:i+5]
        q1_t += q1[i:i+5]
        q2_t += q2[i:i+5]
    
    return (pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L,
            pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t)


def simulate_episodes(groupname, env, Qs, Thetas, gamma=0.9, T=500_000, L=100, n_episodes=1_000):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, '..', '..', 'data', 'simulation_data.h5')
    
    with h5py.File(file_path, 'a') as file:
        if groupname in file:
            msg = "This group already exists!!!"
            raise ValueError(msg)
        else:
            group = file.create_group(groupname)

        # Create datasets within the group to store the variables
        pi1_L_dataset = group.create_dataset('pi1_L', (n_episodes, L), dtype='float')
        pi2_L_dataset = group.create_dataset('pi2_L', (n_episodes, L), dtype='float')
        theta1_L_dataset = group.create_dataset('theta1_L', (n_episodes, L), dtype='float')
        theta2_L_dataset = group.create_dataset('theta2_L', (n_episodes, L), dtype='float')
        q1_L_dataset = group.create_dataset('q1_L', (n_episodes, L), dtype='float')
        q2_L_dataset = group.create_dataset('q2_L', (n_episodes, L), dtype='float')
        pi1_t_dataset = group.create_dataset('pi1_t', (n_episodes, 55), dtype='float')
        pi2_t_dataset = group.create_dataset('pi2_t', (n_episodes, 55), dtype='float')
        theta1_t_dataset = group.create_dataset('theta1_t', (n_episodes, 55), dtype='float')
        theta2_t_dataset = group.create_dataset('theta2_t', (n_episodes, 55), dtype='float')
        q1_t_dataset = group.create_dataset('q1_t', (n_episodes, 55), dtype='float')
        q2_t_dataset = group.create_dataset('q2_t', (n_episodes, 55), dtype='float')
        datasets = [pi1_L_dataset, pi2_L_dataset, theta1_L_dataset, theta2_L_dataset, 
                    q1_L_dataset, q2_L_dataset, pi1_t_dataset, pi2_t_dataset, 
                    theta1_t_dataset, theta2_t_dataset, q1_t_dataset, q2_t_dataset]

        for i in range(n_episodes):
            print(i)
            pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L, \
            pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = \
            simulate_episode_with_regulator(env, Qs, Thetas, gamma)
            # if Qlearning:
            #     pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L, \
            #     pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = \
            #     simulate_episode_Qlearning(env, Qs, Thetas, gamma, T, L)
            # else:
            #     pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L, \
            #     pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t = \
            #     simulate_episode_DQN(env, Qs, Thetas, gamma=gamma, T=T, L=L)

            data = [pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L, pi1_t, pi2_t, theta1_t, theta2_t, q1_t, q2_t]

            for dataset, d in zip(datasets, data):
                dataset[i] = d