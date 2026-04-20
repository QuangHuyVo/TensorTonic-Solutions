def value_iteration_step(values, transitions, rewards, gamma):
    num_states = len(values)
    num_actions = len(transitions[0])
    
    new_values = [0.0] * num_states

    for s in range(num_states):
        best_value = float('-inf')

        for a in range(num_actions):
            expected_value = 0.0

            for s_next in range(num_states):
                prob = transitions[s][a][s_next]
                expected_value += prob * values[s_next]

            q_sa = rewards[s][a] + gamma * expected_value
            best_value = max(best_value, q_sa)

        new_values[s] = best_value

    return new_values