import numpy as np
import sys
import os

class policy_iteration:
    def __init__(self, environment, max_iter=None):
        self.envonment = environment
        if max_iter is None:
            self.max_iter = np.iinfo(np.int32).max
        else:
            self.max_iter = max_iter
        self.d_idx = None
        self.V = np.zeros(len(self.envonment.states))
        self.total_iter = 0

    def get_optimal_policy(self):
        return self.envonment.get_policy(self.envonment.policies[self.d_idx])

    def evaluate_policy(self):
        print('[policy evaluation]')
        if self.d_idx is None:
            self.d_idx = 0
        policy = self.envonment.policies[self.d_idx]
        I = np.eye(len(self.envonment.states))
        transition_prob = self.envonment.get_transition_prob(policy)
        self.V = np.linalg.inv(I - self.envonment.discount_factor * transition_prob)
        self.V = np.matmul(self.V, self.envonment.get_reward(policy))
        print('V:', self.V, end='\n\n')

    def improve_policy(self):
        print('[policy improvement]')
        values = []
        for i, policy in enumerate(self.envonment.policies):
            values.append(list(self.envonment.get_reward(policy) + self.envonment.discount_factor * np.matmul(self.envonment.get_transition_prob(policy), self.V)))
        self.d_idx = np.argmax(np.sum(values, axis=-1))
        print('values:', values, '\n' +\
              'new policy:',  self.envonment.get_policy(self.envonment.policies[self.d_idx]), end='\n\n')

    def fit(self, verbose=True):
        if verbose is False:
            temp = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        for i in range(self.max_iter):
            print('[iter ' + str(i) + ']')
            self.total_iter = i
            prev_idx = self.d_idx
            self.evaluate_policy()
            self.improve_policy()
            if prev_idx == self.d_idx:
                print('[converged]')
                break
        if verbose is False:
            sys.stdout = temp
