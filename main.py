import argparse

from algorithms.environment import *
from algorithms.value_iteration import *
from algorithms.policy_iteration import *
from algorithms.q_learning import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--value_iteration_tolerance', type=float, default=1e-4)

    parser.add_argument('--q_learning_lr', type=float, default=1e-2)
    parser.add_argument('--q_learning_epsilon', type=float, default=0.5)
    parser.add_argument('--q_learning_tolerance', type=float, default=1e-4)
    parser.add_argument('--q_learning_max_iter', type=int, default=10 ** 5)

    parser.add_argument('--verbose', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    env = environment('configs/hw4_config.json')
    # env = environment('configs/two_state_MDP_config.json')
    # env = environment('configs/online_example_config.json')

    args = args()

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(env)

    print('\n[policy iteration]')
    p = policy_iteration(env)
    p.fit(verbose=args.verbose)
    print('total iteration: ', p.total_iter)
    print('optimal policy: ', p.get_optimal_policy())
    print('value: ', p.V)

    print('\n[value iteration]')
    v = value_iteration(env, tolerance=args.value_iteration_tolerance)
    v.fit(verbose=args.verbose)
    print('total iteration: ', v.total_iter)
    print('optimal policy: ', v.get_optimal_policy())
    print('value: ', v.V)

    print('\n[q learning]')
    q = q_learning(env,
                   learning_rate=args.q_learning_lr,
                   epsilon=args.q_learning_epsilon,
                   tolerance=args.q_learning_tolerance,
                   max_iter=args.q_learning_max_iter)
    q.fit(verbose=args.verbose)
    print('total iteration: ', q.total_iter)
    print('optimal policy: ', q.get_optimal_policy())
    print('Q: ', q.Q)


