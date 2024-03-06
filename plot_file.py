import pickle

import numpy as np
import matplotlib.pyplot as plt

with open(f'./baseline_data/Values/theta_list', 'rb') as pkl_wb_obj:
    theta_list = pickle.load(pkl_wb_obj)

with open(f'./baseline_data/Values/value_list', 'rb') as pkl_wb_obj:
    value_list = pickle.load(pkl_wb_obj)

with open(f'./baseline_data/Values/entropy_list', "rb") as pkl_wb_obj:
    last_entropy_list = pickle.load(pkl_wb_obj)

with open(f'./simple_graph_2_data/Values/entropy_2', "rb") as pkl_wb_obj:
    our_last_entropy_list = pickle.load(pkl_wb_obj)
our_last_entropy = our_last_entropy_list[-1]

with open(f'./simple_graph_2_data/Values/value_2', "rb") as pkl_wb_obj:
    value_list_last = pickle.load(pkl_wb_obj)
our_value = value_list_last[-1]

tau_list = np.linspace(0.1, 10.1, 11)

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(7, 5))
plt.plot(tau_list, last_entropy_list, '-g*', label='last_state_opacity')
plt.axhline(y=our_last_entropy, color='k', linestyle='dashed', label=r'$H(Z_T|Y)$ from our method')
plt.xlabel(r"the value of $\tau$")
plt.ylabel(r"baseline comparison: $H(Z_T|Y)$")
plt.title("Baseline comparison: the last-state opacity")
plt.legend(loc='best')
plt.savefig(f'./baseline_data/Graphs/baseline_last_entropy.png')
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(tau_list, value_list, '-r*', label='estimated value')
plt.axhline(y=0.2, color='k', linestyle='dashed', label='value constraint')
plt.xlabel(r"the value of $\tau$")
plt.ylabel(r"baseline comparison: $V(s_0, \theta)$")
plt.title("Baseline comparison: the estimated value")
plt.legend(loc='best')
plt.savefig(f'./baseline_data/Graphs/baseline_value.png')
plt.show()
