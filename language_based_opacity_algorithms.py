import pickle
import time
import numpy as np
from random import choices
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import matplotlib.pyplot as plt

from grid_world_1 import Environment

env = Environment()
GOAL_REWARD = 0.1
GOAL_PENALTY = 0
GAMMA = 0.8  # Discount rate


def get_reward(state, F, goal_reward=GOAL_REWARD):
    (st, q) = state
    if q in F:
        return goal_reward
    else:
        return 0


def pi_theta(state, act, theta):
    # Gibbs policy
    s = env.states.index(state)
    a = env.actions.index(act)
    e_x = np.exp(theta[s, :] - np.max(theta[s, :]))
    return (e_x / e_x.sum(axis=0))[a]


def action_sampler(theta, state):
    prob_list = [pi_theta(state, act, theta) for act in env.actions]
    return choices(env.actions, prob_list, k=1)[0]


def log_policy_gradient(state, act, theta):
    gradient = np.zeros([env.state_size, env.action_size])
    for s_prime in env.state_indices:
        for a_prime in env.action_indices:
            state_p = env.states[s_prime]
            act_p = env.actions[a_prime]
            indicator_s = 0
            indicator_a = 0
            if state == state_p:
                indicator_s = 1
            if act == act_p:
                indicator_a = 1
            partial_pi_theta = indicator_s * (indicator_a - pi_theta(state_p, act_p, theta))
            gradient[s_prime, a_prime] = partial_pi_theta
    return gradient


def P_i_j(state_i, state_j, theta):
    P_theta_i_j = 0
    for act in env.transition[state_i].keys():
        if state_j in env.transition[state_i][act].keys():
            P_theta_i_j += pi_theta(state_i, act, theta) * env.transition[state_i][act][state_j]
    return P_theta_i_j


def nabla_P_i_j(state_i, state_j, theta):
    nabla_theta_P_i_j = np.zeros([env.state_size, env.action_size])
    for act in env.transition[state_i].keys():
        if state_j in env.transition[state_i][act].keys():
            nabla_theta_P_i_j += (env.transition[state_i][act][state_j] *
                                  pi_theta(state_i, act, theta) * log_policy_gradient(state_i, act, theta))
    return nabla_theta_P_i_j


def alpha_t_vec(alpha_tm1, o_t, theta):
    alpha_t = np.zeros(env.state_size)
    for j in env.state_indices:
        state_j = env.states[j]
        b_j_ot = env.emission_function(state_j, o_t)
        for i in env.state_indices:
            state_i = env.states[i]
            P_theta_i_j = P_i_j(state_i, state_j, theta)
            alpha_t[j] += alpha_tm1[i] * P_theta_i_j * b_j_ot
    return alpha_t


def nabla_alpha_t_matrix(alpha_tm1, nabla_alpha_tm1, o_t, theta):
    """
    :param alpha_tm1: the previous alpha: alpha_{t-1}
    :param nabla_alpha_tm1: the previous nabla alpha: nabla_alpha_{t-1}
    :param o_t: the observation at time point t
    :param theta: The parameter for soft-max policy
    :return: nabla_alpha[state_index, [the index for gradient w.r.t theta]]
    """
    nabla_alpha = np.zeros([env.state_size, env.state_size, env.action_size])
    for j in env.state_indices:
        state_j = env.states[j]
        b_j_ot = env.emission_function(state_j, o_t)
        for i in env.state_indices:
            state_i = env.states[i]
            term_1 = P_i_j(state_i, state_j, theta) * b_j_ot * nabla_alpha_tm1[i, :, :]
            term_2 = alpha_tm1[i] * b_j_ot * nabla_P_i_j(state_i, state_j, theta)
            nabla_alpha[j, :, :] = nabla_alpha[j, :, :] + term_1 + term_2
    return nabla_alpha


def get_alpha_and_nabla_alpha(T, y, theta):
    """
    :param T: the length of the trajectory
    :param y: the observation we obtain
    :param theta: the parameter for soft-max policy
    :return: nabla_alpha_T
    The code need optimization. It's quite inefficient now.
    """
    alpha = np.zeros(env.state_size)
    alpha[env.initial_state_idx] = 1
    nabla_alpha = np.zeros([env.state_size, env.state_size, env.action_size])
    for t in range(1, T):
        o_t = y[t]
        nabla_alpha = nabla_alpha_t_matrix(alpha, nabla_alpha, o_t, theta)
        alpha = alpha_t_vec(alpha, o_t, theta)
    return alpha, nabla_alpha


def P_Y(alpha_T):
    return np.sum(alpha_T)


def nabla_P_Y(grad_alpha_T):
    gradient_P_Y = np.zeros([env.state_size, env.action_size])
    for k in env.state_indices:
        gradient_P_Y += grad_alpha_T[k, :, :]
    return gradient_P_Y


def P_q_g_y(q_T, alpha_T, prob_Y):
    prob_q = 0
    for st in env.env_states:
        state = (st, q_T)
        k = env.states.index(state)
        prob_q += alpha_T[k]
    prob_q /= prob_Y
    return prob_q


def nabla_P_q_g_Y(q_T, alpha_T, grad_alpha_T, prob_Y, grad_P_Y):
    grad_P_q_g_Y = np.zeros([env.state_size, env.action_size])
    for st in env.env_states:
        state = (st, q_T)
        k = env.states.index(state)
        grad_P_q_g_Y += grad_alpha_T[k] / prob_Y - alpha_T[k] / (prob_Y ** 2) * grad_P_Y
    return grad_P_q_g_Y


def entropy_a_grad_per_iter(T, y_k, theta):
    alpha_T, grad_alpha_T = get_alpha_and_nabla_alpha(T, y_k, theta)
    p_theta_yk = P_Y(alpha_T)
    grad_P_theta_yk = nabla_P_Y(grad_alpha_T)

    H_per_iter = 0
    nabla_H_per_iter = np.zeros([env.state_size, env.action_size])

    for q_T in env.auto_states:
        p_qT_g_yk = P_q_g_y(q_T, alpha_T, p_theta_yk)
        # estimate the entropy
        log2_p_qT_g_yk = np.log2(p_qT_g_yk) if p_qT_g_yk > 0 else 0
        H_per_iter += p_qT_g_yk * log2_p_qT_g_yk
        # estimate the gradient of entropy
        grad_p_qT_g_yk = nabla_P_q_g_Y(q_T, alpha_T, grad_alpha_T, p_theta_yk, grad_P_theta_yk)
        term_1 = log2_p_qT_g_yk * grad_p_qT_g_yk
        term_2 = p_qT_g_yk * log2_p_qT_g_yk * grad_P_theta_yk
        term_3 = (1/np.log(2)) * grad_p_qT_g_yk
        nabla_H_per_iter += term_1 + term_2 + term_3
    return H_per_iter, nabla_H_per_iter


def entropy_a_grad_multi(T, y_data, theta):
    M = len(y_data)
    H = 0
    nabla_H = np.zeros([env.state_size, env.action_size])
    with ProcessPoolExecutor(max_workers=20) as exe:
        H_a_gradH_list = exe.map(entropy_a_grad_per_iter, repeat(T), y_data, repeat(theta))
    for H_tuple in H_a_gradH_list:
        H += H_tuple[0]
        nabla_H += H_tuple[1]
    H = - H / M
    nabla_H = - nabla_H / M
    return H, nabla_H


def initial_value_approx(s_data, F, gamma=1):
    value_function = 0
    for i in range(s_data.shape[0]):
        total_return = 0
        for t in range(s_data.shape[1]):
            s = s_data[i, t]
            state = env.states[s]
            total_return += gamma ** t * get_reward(state, F)

        value_function += total_return
    value_function = value_function / s_data.shape[0]
    return value_function


def nabla_value_function(theta, s_data, a_data, F, gamma=1):
    value_function_gradient = np.zeros([env.state_size, env.action_size])
    for i in range(s_data.shape[0]):
        log_P_x_i_gradient = np.zeros([env.state_size, env.action_size])
        total_return = 0
        for t in range(s_data.shape[1]):
            s = s_data[i, t]
            a = a_data[i, t]
            state = env.states[s]
            act = env.actions[a]
            log_P_x_i_gradient += log_policy_gradient(state, act, theta)
            total_return += gamma ** t * get_reward(state, F)

        value_function_gradient += total_return * log_P_x_i_gradient
    value_function_gradient = value_function_gradient / s_data.shape[0]
    return value_function_gradient


def sample_data(M, T, theta):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = np.zeros([M, T], dtype=np.int32)
    y_data = []
    for m in range(M):
        y = []
        # start from initial state
        state = env.initial_state
        # Get the observation of initial state
        y.append(env.observation_function_sampler(state))
        # Sample the action from initial state
        act = action_sampler(theta, state)
        for t in range(T):
            s = env.states.index(state)
            s_data[m, t] = s
            a = env.actions.index(act)
            a_data[m, t] = a
            # sample the next state
            state = env.next_state_sampler(state, act)
            # Add the observation
            y.append(env.observation_function_sampler(state))
            # sample action
            act = action_sampler(theta, state)
        y_data.append(y)
    return s_data, a_data, y_data


def value_iterations(threshold, F, gamma=GAMMA):
    """
    :param threshold: threshold for Bellman error
    :param F: The goal states (Please use list)
    :param gamma: discount rate
    :return: value function
    """
    values = np.zeros(env.state_size)
    values_old = np.copy(values)
    Delta = threshold + 0.1
    while Delta > threshold:
        for state in env.states:
            v_n = 0
            for act in env.actions:
                state_p_dict = env.transition[state][act]
                temp_v = 0
                for state_p in state_p_dict.keys():
                    s_p_prob = state_p_dict[state_p]
                    s_p = env.states.index(state_p)
                    temp_v += s_p_prob * (get_reward(state, F) + gamma * values[s_p])
                if temp_v > v_n:
                    v_n = temp_v
            s = env.states.index(state)
            values[s] = v_n
        Delta = np.max(values - values_old)
        values_old = np.copy(values)
    return values


def optimal_policy(opt_values, F, tau=0.01, gamma=GAMMA):
    pi_star = np.zeros([env.state_size, env.action_size])
    for state in env.states:
        for act in env.actions:
            state_p_dict = env.transition[state][act]
            next_v = 0
            for state_p in state_p_dict.keys():
                s_prime_prob = state_p_dict[state_p]
                s_p = env.states.index(state_p)
                next_v += s_prime_prob * (get_reward(state, F) + gamma * opt_values[s_p])
            s = env.states.index(state)
            a = env.actions.index(act)
            pi_star[s, a] = np.exp(next_v / tau) / np.exp(opt_values[s] / tau)
    return pi_star


def extract_opt_theta(opt_values, F, tau=0.01):
    pi_star = optimal_policy(opt_values, F, tau)
    theta = np.log(pi_star)
    return theta


def main():
    ex_num = 1
    # Define hyperparameters
    iter_num = 200  # iteration number of gradient ascent
    M = 100  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 1  # step size for theta
    kappa = 0.2  # constant step size for lambda
    F = [1]  # Define the goal automaton state
    alpha = 0.1  # value constraint
    # Initialize the parameters
    theta = np.random.random([env.state_size, env.action_size])
    # opt_values = value_iterations(1e-3, F)
    # theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    lam = np.random.uniform(1, 10)
    # Create empty lists
    entropy_list = []
    value_list = []
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, a_data, y_data = sample_data(M, T, theta)
        # Gradient ascent process
        print(y_data)
        approx_entropy, grad_H = entropy_a_grad_multi(T, y_data, theta)
        # print("The gradient of entropy is", grad_H)
        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        # SGD gradients
        grad_V = nabla_value_function(theta, s_data, a_data, F)
        # print("The gradient of value function is", grad_V)
        grad_L = grad_H + lam * grad_V
        approx_value = initial_value_approx(s_data, F)
        value_list.append(approx_value)
        print("The estimated value is", approx_value)
        # SGD updates
        theta = theta + eta * grad_L
        print("The lambda is", lam)
        # kappa = 1 / (i + 1) # changed step size
        lam = lam - kappa * (approx_value - alpha)
        ###############################################
        end = time.time()
        print("One iteration done. It takes", end - start, "s")

    with open(f'./grid_world_1_data/Values/theta_{ex_num}', 'wb') as f:
        np.save(f, theta)

    with open(f'./grid_world_1_data/Values/entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    with open(f'./grid_world_1_data/Values/value_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(value_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label='entropy')
    plt.plot(iteration_list, value_list, label='estimated value')
    plt.xlabel("The iteration number")
    plt.ylabel("entropy and value")
    plt.legend()
    plt.savefig(f'./grid_world_1_data/Graphs/Ex_{ex_num}.png')
    plt.show()


if __name__ == "__main__":
    main()
