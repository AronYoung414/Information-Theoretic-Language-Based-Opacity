import numpy as np
import pickle
import matplotlib.pyplot as plt
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


def estimate_prob_error(M, T, theta, s_data, y_data):
    error_counter = 0
    for k in range(M):
        s_k = s_data[k]
        y_k = y_data[k]
        true_state = env.states[s_k[-1]][1]
        p_theta_q_g_y = ML_estimate(T, y_k, theta)
        ML_estimator = np.argmax(p_theta_q_g_y)
        if true_state != ML_estimator:
            error_counter += 1
    prob_error = error_counter / M
    return prob_error


def main():
    M = 1000
    T = 5
    tau_list = np.linspace(0.1, 20.1, 11)
    # estimate the prob of error for our method
    with open(f'./simple_graph_2_data/Values/theta_2', "rb") as pkl_wb_obj:
        theta = np.load(pkl_wb_obj)
    s_data, a_data, y_data = sample_data(M, T, theta)
    our_prob_error = estimate_prob_error(M, T, theta, s_data, y_data)
    print("Our probability of error is", our_prob_error)
    # estimate the prob of error for entropy regularized MDP of different tau
    prob_error_list_reg = []
    with open(f'./baseline_data/Values/theta_list_2', "rb") as pkl_wb_obj:
        theta_list = pickle.load(pkl_wb_obj)
    for theta in theta_list:
        s_data, a_data, y_data = sample_data(M, T, theta)
        prob_error = estimate_prob_error(M, T, theta, s_data, y_data)
        prob_error_list_reg.append(prob_error)
        print("One evaluation done. The prob of error is", prob_error)

    plt.rcParams.update({'font.size': 15})

    plt.figure(figsize=(7, 5))
    plt.plot(tau_list, prob_error_list_reg, '-g*', label='Entropy-regularized MDP')
    plt.axhline(y=our_prob_error, color='k', linestyle='dashed', label=r'Our method')
    plt.xlabel(r"the value of $\tau$")
    plt.ylabel(r"the guess error")
    plt.title("Baseline comparison: the guess error")
    plt.legend(loc='best')
    plt.savefig(f'./baseline_data/Graphs/guess_error_2.png')
    plt.show()


if __name__ == "__main__":
    main()
