import numpy as np
import json
import itertools
import os

class environment:
    def __init__(self, filename):
        if os.path.isfile(filename):
            config = json.load(open(filename, 'r'))
            self.states = config['states']
            self.actions = config['actions']
            self.rewards = np.array(config['rewards'])
            self.discount_factor = config['discount_factor']
            self.transition_prob_per_action = np.array(config['transition_prob_per_action'])
            # print(config)

        self.action_space = self.transition_prob_per_action.sum(axis=-1) != 0

        state_list, action_list = np.where(self.action_space.T == True)
        self.state_action_list = []

        for i, state in enumerate(self.states):
            self.state_action_list.append(action_list[np.where(state_list == i)])
        self.policies = list(itertools.product(*self.state_action_list))

    def __str__(self):
        out = "[environment setting]" + "\n" +\
              "====================================================================" + "\n" +\
              "states: " + str(self.states) + "\n" +\
              "actions: " + str(self.actions) + "\n" +\
              "discount_factor: " + str(self.discount_factor) + "\n" +\
              "policies: " + str([list(np.array(self.actions)[list(p)]) for p in self.policies]) + "\n" +\
              "====================================================================" + "\n"
        return out

    def get_transition_prob(self, policy):
        transition_prob = np.zeros([len(self.states), len(self.states)])
        for i, state in enumerate(self.states):
            action = policy[i]
            transition_prob[i, :] = self.transition_prob_per_action[action, i, :]
        return transition_prob

    def get_reward(self, policy):
        reward = np.zeros(len(self.states))
        for i, state in enumerate(self.states):
            action = policy[i]
            reward[i] = self.rewards[action, i]
        return reward

    def get_policy(self, policy):
        return list(zip(self.states, np.array(self.actions)[list(policy)]))
