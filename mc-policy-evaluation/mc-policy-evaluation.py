import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0.0
        first_visit_returns = [0.0] * len(episode)

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            first_visit_returns[t] = G

        visited = set()
        for t, (state, reward) in enumerate(episode):
            if state not in visited:
                returns[state].append(first_visit_returns[t])
                visited.add(state)

    V = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])

    return V
