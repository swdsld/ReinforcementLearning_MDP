import numpy as np
import sys
import os

INF = np.iinfo(np.int32).max

class value_iteration:
    def __init__(self, environment, tolerance, max_iter=None):
        self.envonment = environment
        self.tolerance = tolerance
        if max_iter is None:
            self.max_iter = np.iinfo(np.int32).max
        else:
            self.max_iter = max_iter
        self.V = np.zeros(len(self.envonment.states))
        self.total_iter = 0
        self.optimal_policy = None

    def get_optimal_policy(self):
        return self.envonment.get_policy(self.optimal_policy)

    def update_value(self):
        print('[update_value]')
        values = []
        for i, action in enumerate(
                self.envonment.actions):
            values.append(self.envonment.rewards[i] + self.envonment.discount_factor * np.matmul(self.envonment.transition_prob_per_action[i], self.V))
        values = np.array(values)
        values[np.where(self.envonment.action_space == False)] = -INF
        self.V = np.max(values, axis=0)

        print('V:', self.V, end='\n\n')

    def retrieve_policy(self):
        print('[policy retrieval]')
        values = []
        for i, action in enumerate(
                self.envonment.actions):
            values.append(self.envonment.rewards[i] + self.envonment.discount_factor * np.matmul(
                self.envonment.transition_prob_per_action[i], self.V))
        values = np.array(values)
        values[np.where(self.envonment.action_space == False)] = -INF
        self.optimal_policy = np.argmax(values, axis=0)
        print('values:', values, end='\n\n')

    def fit(self, verbose=True):
        if verbose is False:
            temp = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        for i in range(self.max_iter):
            print('[iter ' + str(i) + ']')
            self.total_iter = i
            prev_value = np.copy(self.V)
            self.update_value()

            if np.linalg.norm(prev_value - self.V) < self.tolerance * ((1 - self.envonment.discount_factor) / (2 * self.envonment.discount_factor)):
                print('[converged]')
                break
        self.retrieve_policy()
        print('optimal policy: ', self.get_optimal_policy())
        if verbose is False:
            sys.stdout = temp