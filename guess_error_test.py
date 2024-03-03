import numpy as np
from language_based_opacity_algorithms import sample_data
from language_based_opacity_algorithms import P_q_g_y
from language_based_opacity_algorithms import get_alpha_and_nabla_alpha
from language_based_opacity_algorithms import P_Y
from language_based_opacity_algorithms import value_iterations
from language_based_opacity_algorithms import extract_opt_theta

from simple_graph_2 import Environment

env = Environment()


def ML_estimate(T, y_k, theta):
    alpha_T, grad_alpha_T = get_alpha_and_nabla_alpha(T, y_k, theta)
    prob_Y = P_Y(alpha_T)
    prob_q_vec = []
    for q_T in env.auto_states:
        prob_q = P_q_g_y(q_T, alpha_T, prob_Y)
        prob_q_vec.append(prob_q)
    return np.array(prob_q_vec)


def main():
    M = 100
    T = 5
    # theta = np.random.random([env.state_size, env.action_size])
    F = env.goals  # Define the goal automaton state
    opt_values = value_iterations(1e-3, F)
    theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    s_data, a_data, y_data = sample_data(M, T, theta)
    with open(f'./simple_graph_2_data/Values/theta_2', "rb") as pkl_wb_obj:
        theta = np.load(pkl_wb_obj)
    error_counter = 0
    for k in range(len(s_data)):
        s_k = s_data[k]
        y_k = y_data[k]
        true_state = env.states[s_k[-1]][1]
        p_theta_q_g_y = ML_estimate(T, y_k, theta)
        ML_estimator = np.argmax(p_theta_q_g_y)
        if true_state != ML_estimator:
            error_counter += 1
    prob_error = error_counter / M
    print(prob_error)


if __name__ == "__main__":
    main()
