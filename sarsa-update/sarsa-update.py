def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    current_q = q_table[state][action]
    next_q = q_table[next_state][next_action]
    
    q_table[state][action] = current_q + alpha * (reward + gamma * next_q - current_q)
    
    return q_table