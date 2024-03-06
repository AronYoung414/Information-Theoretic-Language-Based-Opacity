import numpy as np
import pickle
import matplotlib.pyplot as plt
from language_based_opacity_algorithms import sample_data
from language_based_opacity_algorithms import initial_value_approx
from language_based_opacity_algorithms import extract_opt_theta
from language_based_opacity_algorithms import entropy_a_grad_multi

from simple_graph_2 import Environment

env = Environment()
GOAL_REWARD = 0.1
GAMMA = 0.8  # Discount rate


def entropy_value_iterations(threshold, tau, gamma=GAMMA):
    values = np.zeros(env.state_size)
    values_old = np.copy(values)
    Delta = threshold + 0.1
    while Delta > threshold:
        for state in env.states:
            v_n = 0
            for act in env.actions:
                state_p_dict = env.transition[state][act]
                next_v = 0
                for state_p in state_p_dict.keys():
                    s_p_prob = state_p_dict[state_p]
                    s_p = env.states.index(state_p)
                    next_v += s_p_prob * (env.get_reward(state) + gamma * values[s_p])
                temp_v = tau * np.log(env.action_size * np.exp(next_v/tau))
                if temp_v > v_n:
                    v_n = temp_v
            s = env.states.index(state)
            values[s] = v_n
        Delta = np.max(values - values_old)
        values_old = np.copy(values)
    return values


F = env.goals
tau_list = np.linspace(0.1, 20.1, 11)
print(tau_list)

M = 100  # number of sampled trajectories
T = 10  # length of a trajectory
theta_list = []
value_list = []
last_entropy_list = []
initial_entropy_list = []
for tau in tau_list:
    opt_values = entropy_value_iterations(1e-3, tau)
    theta = extract_opt_theta(opt_values, tau)
    theta_list.append(theta)
    s_data, a_data, y_data = sample_data(M, T, theta)
    approx_value = initial_value_approx(s_data)
    value_list.append(approx_value)
    print("The estimated value is", approx_value)

    approx_entropy, grad_H = entropy_a_grad_multi(T, y_data, theta)
    last_entropy_list.append(approx_entropy)
    print("The last-state opacity is", approx_entropy)

    with open(f'./baseline_data/Values/theta_list_2', 'wb') as pkl_wb_obj:
        pickle.dump(theta_list, pkl_wb_obj)

    with open(f'./baseline_data/Values/value_list_2', 'wb') as pkl_wb_obj:
        pickle.dump(value_list, pkl_wb_obj)

    with open(f'./baseline_data/Values/entropy_list_2', "wb") as pkl_wb_obj:
        pickle.dump(last_entropy_list, pkl_wb_obj)


plt.plot(tau_list, last_entropy_list, label='last_state_opacity')
plt.plot(tau_list, value_list, label='estimated value')
plt.xlabel(r"The value of $\tau$")
plt.ylabel("entropy and value")
plt.legend()
plt.savefig(f'./baseline_data/Graphs/baseline_2.png')
plt.show()
