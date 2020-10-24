from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3  # threshold for convergence (in value table)


def print_values(V, g):
    """ Function to visualise values - prints the value of each state on top of drawing of environment
    V: value table
    g: gridworld object
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            # If value is not in dictionary, use default value 0
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    """ Prints action for each state on top of drawing of environment.
    P: policy table
    g: gridworld object
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


if __name__ == '__main__':

    """
    ### define transition probabilities and grid ###
    # Most general case is probs[(s, a, s', r)] = p(s', r |s, a). But we assume reward deterministic.
    # Then transition_probs[(s, a, s')] = p(s' | s, a)
    # So in dict form, key is (s, a, s'), the value is the probability
    # any key NOT present will considered to be impossible (i.e. probability 0)
    """
    transition_probs = {}

    """
    # To reduce the dimensionality of the dictionary, we'll use deterministic rewards, r(s, a, s')
    # You could make it simpler by using r(s') since the reward doesn't actually depend on (s, a)
    # Below we initiate a grid object, loop through the cells and if the cell (i.e. state)
     is not terminal, we fetch the next state, transition prob and reward, i.e. s', p(s, a, s'), r(s')
    """
    rewards = {}
    # Fetching the grid is akin to creating the environment
    grid = standard_grid()
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    """ Represent policy as dict, with key being the state, value being the action.
     # we have a fixed policy, which we can evaluate. 
    """
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    print_policy(policy, grid)

    # initialize V(s) = 0
    V = {}
    for s in grid.all_states():
        V[s] = 0

    gamma = 0.9  # discount factor

    """ Policy evaluation Code - Main Purpose of Script"""
    # repeat until convergence
    it = 0
    # Keep looping while max change of V(s) is greater than threshold set
    while True:
        biggest_change = 0
        # Loop through all states
        for s in grid.all_states():
            # If state is terminal, we already known V(s) = 0
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # we will accumulate the answer
                # Loop through action space to get new_v
                for a in ACTION_SPACE:
                    # Another loop through state space, but this time for s2
                    for s2 in grid.all_states():
                        # action probability is deterministic so action_prob = 1 or 0.
                        action_prob = 1 if policy.get(s) == a else 0
                        # reward is a function of (s, a, s'), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        # Bellmans equation: V(s) = sum(action prob * transition prob * (r  +yV(s')
                        # Note summation is done over a and s'
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

                # after done getting the new value, update the value table
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        print("iter:", it, "biggest_change:", biggest_change)
        print_values(V, grid)
        it += 1

        if biggest_change < SMALL_ENOUGH:
            break
    print("\n\n")
