from random import choices
import numpy as np


class Environment:

    def __init__(self):
        # parameter which controls observation noise
        self.obs_noise = 0.5
        # Define states
        self.env_states = ["0", "1", "2", "3", "4", "h1", "h2"]
        self.auto_states = [0, 1, 2, 3]
        # Goals
        self.goals = ["2", "4"]  # G0 and G1 of environment states
        self.auto_goals = [3]
        # Product space
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
        # Secrets
        self.secrets = ["h1", "h2"]
        # transition probability dictionary
        self.env_transition = self.get_env_transition()
        self.auto_transition = self.get_auto_transition()
        self.transition = self.get_transition()
        # Define observations
        self.observations = ["0", "r", "b", "3"]
        # Define sensors
        self.sensors = [["0"],
                        ["h1"],
                        ["2"],
                        ["1", "4"],
                        ["h2", "3"]]

    def get_reward(self, state):
        F = self.goals
        (st, q) = state
        if st == F[0]:
            return 1
        elif st == F[1]:
            return 0.1
        else:
            return 0

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
        trans = {"0": {"a1": {"h1": 1}, "a2": {"1": 1}},
                 "1": {"a1": {"3": 0.5, "4": 0.5}, "a2": {"h2": 1}},
                 "2": {"a1": {"3": 1}, "a2": {"h1": 1}},
                 "3": {"a1": {"1": 0.5, "h2": 0.5}, "a2": {"1": 0.5, "h2": 0.5}},
                 "4": {"a1": {"1": 0.5, "h2": 0.5}, "a2": {"3": 1}},
                 "h1": {"a1": {"2": 1}, "a2": {"2": 1}},
                 "h2": {"a1": {"3": 0.5, "4": 0.5}, "a2": {"3": 0.5, "4": 0.5}}
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
            return ["h1"]
        elif st in self.sensors[2]:
            return ["2"]
        elif st in self.sensors[3]:
            return ["b", "n"]
        else:
            return ["r", "n"]

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
