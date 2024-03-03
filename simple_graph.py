from random import choices
import numpy as np


class Environment:

    def __init__(self):
        # parameter which controls observation noise
        self.obs_noise = 0.5
        # Define states
        self.env_states = ["0", "1", "2", "3", "4", "h1", "h2"]
        self.auto_states = [0, 1, 2, 3]
        self.auto_goals = [3]
        self.states = [(env, auto) for env in self.env_states for auto in self.auto_states]
        self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)
        # Define initial state
        self.initial_state = ("0", 0)
        self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.actions = ["a1", "a2"]
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # Goals
        self.goals = ["3"]  # G0 and G1 of environment states
        # Secrets
        self.secrets = ["h1"]
        # transition probability dictionary
        self.env_transition = self.get_env_transition()
        self.auto_transition = self.get_auto_transition()
        self.transition = self.get_transition()
        # Define observations
        self.observations = ["0", "r", "b", "3", "n"]
        # Define sensors
        self.sensors = [["0"],
                        ["1", "4"],
                        ["h1", "h2", "2"],
                        ["3"]]

    def label_function(self, env_state):
        if env_state in self.goals:
            return "g"
        elif env_state in self.secrets:
            return "s"
        else:
            return "n"

    def get_auto_transition(self):
        trans = {0: {"g": 1, "s": 2, "n": 0},
                 1: {"g": 1, "s": 2, "n": 1},
                 2: {"g": 3, "s": 2, "n": 2},
                 3: {"g": 3, "s": 3, "n": 3}}
        return trans

    def get_env_transition(self):
        # Constructing transition function trans[state][action][next_state] = probability
        trans = {"0": {"a1": {"h1": 0.3, "1": 0.7}, "a2": {"4": 1}},
                 "1": {"a1": {"2": 0.7, "h2": 0.3}, "a2": {"2": 0.7, "h2": 0.3}},
                 "2": {"a1": {"3": 1}, "a2": {"h1": 1}},
                 "3": {"a1": {"0": 0.5, "4": 0.5}, "a2": {"2": 0.5, "h2": 0.5}},
                 "4": {"a1": {"1": 1}, "a2": {"1": 1}},
                 "h1": {"a1": {"2": 0.5, "h2": 0.5}, "a2": {"2": 0.5, "h2": 0.5}},
                 "h2": {"a1": {"3": 1}, "a2": {"1": 1}}
                 }
        return trans

    def get_transition(self):
        trans = {}
        for state in self.states:
            trans[state] = {}
            for act in self.actions:
                trans[state][act] = {}
                (st, q) = state
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
            return ["0"]
        elif st in self.sensors[1]:
            return ["r", "n"]
        elif st in self.sensors[2]:
            return ["b", "n"]
        else:
            return ["3"]

    def observation_function_sampler(self, state):
        observation_set = self.observation_function(state)
        if len(observation_set) > 1:
            return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]
        else:
            return observation_set[0]

    def emission_function(self, state, o):
        observation_set = self.observation_function(state)
        if o in observation_set:
            if len(observation_set) == 1:
                return 1
            else:
                if o == "n":
                    return self.obs_noise
                else:
                    return 1 - self.obs_noise
        else:
            return 0
