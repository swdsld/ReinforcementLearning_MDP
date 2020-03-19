import numpy as np
import sys
import os

INF = np.iinfo(np.int32).max

class q_learning:
    def __init__(self, environment, learning_rate, epsilon, tolerance, max_iter=None):
        self.envonment = environment
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.tolerance = tolerance
        if max_iter is None:
            self.max_iter = np.iinfo(np.int32).max
        else:
            self.max_iter = max_iter
        self.Q = np.zeros([len(self.envonment.states), len(self.envonment.actions)])
        self.Q[np.where(self.envonment.action_space.T == False)] = -INF
        self.current_state = 0
        self.optimal_policy = None
        self.total_iter = 0

    def get_optimal_policy(self):
        return self.envonment.get_policy(self.optimal_policy)

    def update_Q(self, learning_rate):
        print('[update_Q]')

        # dropping to random state since there are some cases with trap state which we cannot escape without this
        if np.random.choice(['explore', 'exploit'], p=[self.epsilon, 1 - self.epsilon]) == 'explore':
            self.current_state = np.random.choice([i for i in range(len(self.envonment.states))])

        if np.random.choice(['explore', 'exploit'], p=[self.epsilon, 1 - self.epsilon]) == 'explore':
            action = np.random.choice(self.envonment.state_action_list[self.current_state])
        else:
            action = np.argmax(self.Q[self.current_state]) # without exploration
        reward = self.envonment.rewards[action, self.current_state]

        state = [i for i in range(len(self.envonment.states))]
        next_state = np.random.choice(state, p=self.envonment.transition_prob_per_action[action][self.current_state])
        q_tilde = reward + self.envonment.discount_factor * np.max(self.Q[next_state])

        self.Q[self.current_state, action] = (1 - learning_rate) * self.Q[self.current_state, action] + learning_rate * q_tilde

        print('state:', self.current_state, 'action:', action)
        self.current_state = next_state
        print('Q:', self.Q, end='\n\n')

    def retrieve_policy(self):
        print('[policy retrieval]')
        self.optimal_policy = np.argmax(self.Q, axis=1)
        states = [i for i in range(len(self.envonment.states))]
        print('Q values:', self.Q[states, self.optimal_policy], end='\n\n')

    def fit(self, verbose=True):
        if verbose is False:
            temp = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        for i in range(self.max_iter):
            print('[iter ' + str(i) + ']')
            self.total_iter = i
            prev_Q = np.copy(self.Q)
            self.update_Q(self.learning_rate)# * ((0.99) ** i))
            if np.linalg.norm(prev_Q - self.Q, np.inf) < self.tolerance and i > self.max_iter // 10:
                print('[converged]')
                break
        self.retrieve_policy()
        print('optimal policy: ', self.get_optimal_policy())
        if verbose is False:
            sys.stdout = temp