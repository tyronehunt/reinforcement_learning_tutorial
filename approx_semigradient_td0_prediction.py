from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import play_game, SMALL_ENOUGH, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS


# NOTE: this is only policy evaluation, not optimization

class Model:
    """
    Now giving the model a class with an api. We initialise a random-normal array theta, of length 4.
    """
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self, s):
        # Transforms state into feature vector (may not be a good feature!)
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])

    def predict(self, s):
        # Takes in a state, transforms it to feature x and does dot product with theta. I.e. predict Vh.
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        # Just returns x, for a given state.
        return self.s2x(s)


if __name__ == '__main__':
    # use the standard grid again (0 for every step) so that we can compare to iterative policy evaluation
    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    model = Model()
    deltas = []

    # repeat until convergence
    k = 1.0
    for it in range(20000):
        if it % 10 == 0:
            k += 0.01
        # Decaying learning rate
        alpha = ALPHA / k
        biggest_change = 0

        # generate an episode using policy, pi
        states_and_rewards = play_game(grid, policy)
        # the first (s, r) tuple is the state we start in and 0
        # the last (s, r) tuple is the terminal state and the final reward = 0, so don't care about updating it.

        # Loop through states and rewards
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]
            # we will update V(s) AS we experience the episode
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA * model.predict(s2)
            model.theta += alpha * (target - model.predict(s)) * model.grad(s)
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # obtain predicted values
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            # terminal state or state we can't otherwise get to
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
