from classes.action import Action
from classes.agent import Agent
import numpy as np


def create_actions(Qs, k):
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
    Thetas = [theta/k for theta in range(k+1)]
    actions_h = [Action(q, theta) for q in Qs for theta in Thetas]
    actions_h.append(Action(0, 1))
    actions = np.array(actions_h)
    return actions


def simulate_episode(env, Qs, k=6, gamma=0.9, T=500_000, L=1_000):
    """
    Simulates an episode of the game.

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
    actions = create_actions(Qs, k)

    # Pick the first two prices of both players randomly
    Action1, Action2, Action1_next, Action2_next = np.random.choice(actions, size=4)
    
    # Create both agents
    Agent1 = Agent(actions, gamma)
    Agent2 = Agent(actions, gamma)

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
    
    return (pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)
