from random import choices
import numpy as np

WIDTH = 6
LENGTH = 6


class Environment:

    def __init__(self):
        # parameter which controls environment noise
        self.stoPar = 0.1
        # parameter which controls observation noise
        self.obs_noise = 1 / 3
        # Define states
        self.env_states = [(i, j) for i in range(WIDTH) for j in range(LENGTH)]
        self.auto_states = [0, 1, 2, 3]
        self.auto_goals = [1, 3]
        self.states = [(env, auto) for env in self.env_states for auto in self.auto_states] + [('sink', 'sink')]
        self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)
        # Define initial state
        self.initial_state = ((3, 1), 0)
        self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # Goals
        self.goals = [(0, 2), (2, 4)]  # G0 and G1 of environment states
        # Secrets
        self.secrets = [(5, 2)]
        # transition probability dictionary
        self.env_transition = self.get_env_transition()
        self.auto_transition = self.get_auto_transition()
        self.transition = self.get_transition()
        # Define observations
        self.observations = ['1', '2', '3', '0']
        # Define sensors
        self.sensors = [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                        [(3, 1), (4, 0), (4, 1), (4, 2), (5, 1), (5, 2)],
                        [(1, 4), (2, 4), (2, 5), (3, 4), (3, 5), (4, 4), (4, 5), (5, 5)]]

    def label_function(self, env_state):
        if env_state in self.goals:
            return "g"
        elif env_state in self.secrets:
            return "s"
        else:
            return "n"

    def get_auto_transition(self):
        trans = {0: {'g': 1, 's': 2, "n": 0},
                 1: {'g': 1, 's': 1, "n": 1},
                 2: {'g': 3, 's': 2, "n": 2},
                 3: {'g': 3, 's': 3, "n": 3}}
        return trans

    def complementary_actions(self, act):
        # Use to find out stochastic transitions, if it stays, no stochasticity, if other actions, return possible stochasticity directions.
        if act == (0, 0):
            return []
        elif act[0] == 0:
            return [(1, 0), (-1, 0)]
        else:
            return [(0, 1), (0, -1)]

    def check_inside(self, st):
        # If the state is valid or not
        if st in self.env_states:
            return True
        return False

    def get_env_transition(self):
        # Constructing transition function trans[state][action][next_state] = probability
        stoPar = self.stoPar
        trans = {}
        for st in self.env_states:
            trans[st] = {}
            for act in self.actions:
                if act == (0, 0):
                    trans[st][act] = {}
                    trans[st][act][st] = 1
                else:
                    trans[st][act] = {}
                    trans[st][act][st] = 0
                    tempst = tuple(np.array(st) + np.array(act))
                    if self.check_inside(tempst):
                        trans[st][act][tempst] = 1 - 2 * stoPar
                    else:
                        trans[st][act][st] += 1 - 2 * stoPar
                    for act_ in self.complementary_actions(act):
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.check_inside(tempst_):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
        # self.check_trans(trans)
        return trans

    def get_transition(self):
        trans = {}
        for state in self.states:
            trans[state] = {}
            for act in self.actions:
                trans[state][act] = {}
                (st, q) = state
                if q in self.auto_goals:
                    next_state = ('sink', 'sink')
                    trans[state][act][next_state] = 1
                elif q == 'sink':
                    next_state = ('sink', 'sink')
                    trans[state][act][next_state] = 1
                else:
                    for next_st in self.env_transition[st][act].keys():
                        next_q = self.auto_transition[q][self.label_function(st)]
                        next_state = (next_st, next_q)
                        trans[state][act][next_state] = self.env_transition[st][act][next_st]
        self.check_trans(trans)
        return trans

    def check_trans(self, trans):
        # Check if the transitions are constructed correctly
        for st in trans.keys():
            for act in trans[st].keys():
                if abs(sum(trans[st][act].values()) - 1) > 0.01:
                    print("st is:", st, "act is:", act, "sum is:", sum(self.transition[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    def next_state_sampler(self, state, act):
        next_supp = list(self.transition[state][act].keys())
        next_prob = [self.transition[state][act][next_s] for next_s in next_supp]
        next_state = choices(next_supp, next_prob)[0]
        return next_state

    def observation_function(self, state):
        (st, q) = state
        if st in self.sensors[0]:
            return ['1', '0']
        elif st in self.sensors[1]:
            return ['2', '0']
        elif st in self.sensors[2]:
            return ['3', '0']
        else:
            return ['s']  # sink state

    def observation_function_sampler(self, state):
        observation_set = self.observation_function(state)
        if len(observation_set) > 1:
            return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]
        else:
            return self.observations[-1]

    def emission_function(self, state, o):
        observation_set = self.observation_function(state)
        if o in observation_set:
            if len(observation_set) == 1:
                return 1
            else:
                if o == '0':
                    return self.obs_noise
                else:
                    return 1 - self.obs_noise
        else:
            return 0
