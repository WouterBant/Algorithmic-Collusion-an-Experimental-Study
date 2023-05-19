from classes import *


def create_actions(Qs, k):
    Thetas = [theta/k for theta in range(k+1)]
    actions_h = [Action(q, theta) for q in Qs for theta in Thetas]
    actions_h.append(Action(0, 1))
    actions = np.array(actions_h)
    return actions


def Qlearn(env, Qs, k=6, gamma=0.9, T=500_000, L=1_000):
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
            # Calculate the current profits
            pi1, pi2 = env.Pi(Action1_next, Action2)

            # Add q and theta
            q1.append(Action1_next.q)
            theta1.append(Action1_next.theta)

            # Calculate the next profit
            pi1_next, _ = env.Pi(Action1_next, Action2_next)

            # Update Q-function
            Agent1.learn(state=Action2, action=Action1_next, 
                         next_state=Action2_next, profit=pi1, 
                         next_profit=pi1_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action1, Action1_next = Action1_next, Agent1.act(state=Action2_next, time=t)
        else:
            # Calculate the current profits
            pi1, pi2 = env.Pi(Action1, Action2_next)

            # Add q and theta
            q2.append(Action2_next.q)
            theta2.append(Action2_next.theta)

            # Calculate the next profit
            _, pi2_next = env.Pi(Action1_next, Action2_next)

            # Update Q-function
            Agent2.learn(state=Action1, action=Action2_next,
                         next_state=Action1_next, profit=pi2,
                         next_profit=pi2_next, time=t)

            # Update current action player 1, and determine next action based on the next state
            Action2, Action2_next = Action2_next, Agent2.act(state=Action1_next, time=t)
        
        # Append the profits to the lists of profits
        pi1_ep.append(pi1)
        pi2_ep.append(pi2)
    
    # Get the profits, theta's, and q's from the last L runs
    pi1_L, pi2_L = pi1_ep[-L:], pi2_ep[-L:]
    theta1_L, theta2_L = theta1[-L:], theta2[-L:]
    q1_L, q2_L = q1[-L:], q2[-L:]
    
    return (pi1_L, pi2_L, theta1_L, theta2_L, q1_L, q2_L)
