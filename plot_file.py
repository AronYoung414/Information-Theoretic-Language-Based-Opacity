import pickle

import numpy as np
import matplotlib.pyplot as plt


with open(f'./grid_world_1_data/Values/value_4', 'rb') as pkl_wb_obj:
    value_list_4 = pickle.load(pkl_wb_obj)

with open(f'./grid_world_1_data/Values/entropy_4', "rb") as pkl_wb_obj:
    entropy_list_4 = pickle.load(pkl_wb_obj)

with open(f'./grid_world_1_data/Values/value_5', 'rb') as pkl_wb_obj:
    value_list_5 = pickle.load(pkl_wb_obj)

with open(f'./grid_world_1_data/Values/entropy_5', "rb") as pkl_wb_obj:
    entropy_list_5 = pickle.load(pkl_wb_obj)

value_list = value_list_4 + value_list_5
entropy_list = entropy_list_4 + entropy_list_5

iter_num = 1300  # iteration number of gradient ascent
iteration_list = range(iter_num)
plt.plot(iteration_list, entropy_list, label='entropy')
plt.plot(iteration_list, value_list, label='estimated value')
plt.xlabel("The iteration number")
plt.ylabel("entropy and value")
plt.legend()
plt.savefig(f'./grid_world_1_data/Graphs/Ex_45.png')
plt.show()
