import numpy as np
from language_based_opacity_algorithms import sample_data
from grid_world_1 import Environment
env = Environment()

# print(env.transition)

theta = np.random.random([env.state_size, env.action_size])
s_data, a_data, y_data = sample_data(1, 100, theta)
for i in range(len(s_data[0])):
    print("The state is", env.states[s_data[0][i]])
    print("The action is", env.actions[a_data[0][i]])
    print("The observation is", y_data[0][i])
