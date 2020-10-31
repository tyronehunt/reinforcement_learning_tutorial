from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

""" This file is back to deterministic version of gridworld, i.e. state transitions are deterministic. """


def get_transition_probs_and_rewards(grid):
    """ Populate state transition probs and rewards
  # copied from previous: iterative_policy_evaluation work (but in function form)
  """
    ### define transition probabilities and grid ###
    # the key is (s, a, s'), the value is the transition probability, i.e. transition_probs[(s, a, s')] = p(s' | s, a)
    # Any key NOT present will considered to be impossible (i.e. probability 0)
    transition_probs = {}

    # to reduce the dimensionality of the dictionary, we'll use deterministic rewards, r(s, a, s')
    # note: you could make it simpler by using r(s') since the reward doesn't actually depend on (s, a)
    rewards = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    return transition_probs, rewards


def evaluate_deterministic_policy(grid, policy):
    """ This evaluates V(s) given an existing policy. It's the previous evaluate.py files in functional format.
    Param: gridworld object (environment)
    Param: policy dictionary
    # Could pass in transition probs and rewards dict - but just fetching them globally.
    """
    # initialize V(s) = 0
    V = {}
    for s in grid.all_states():
        V[s] = 0

    # repeat until convergence
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # we will accumulate the answer
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        # action probability is deterministic
                        action_prob = 1 if policy.get(s) == a else 0

                        # reward is a function of (s, a, s'), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                # after done getting the new value, update the value table
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        it += 1

        if biggest_change < SMALL_ENOUGH:
            break
    return V


if __name__ == '__main__':

    # Get grid world object
    grid = standard_grid()
    # Get the transition probs and rewards
    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # print rewards - takes in generic dictionary where key is state and value is number.
    # Thus although function originally to print values, it can also print rewards as same dict structure.
    print("rewards:")
    print_values(grid.rewards, grid)

    # Make a random policy: we'll randomly choose an action and update as we learn
    policy = {}
    # Loop through non-terminal states
    for s in grid.actions.keys():
        # for each state -> assign a random action
        policy[s] = np.random.choice(ACTION_SPACE)

    # initial policy
    print("initial policy:")
    print_policy(policy, grid)

    # repeat until convergence - will break out when policy does not change
    while True:

        # policy evaluation step - we already know how to do this!
        V = evaluate_deterministic_policy(grid, policy)

        # policy improvement step
        is_policy_converged = True
        # For non-terminal states that we can perform actions:
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            # We will incrementally increase this best value
            best_value = float('-inf')

            # loop through all possible actions to find the best current action
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    # reward is a function of (s, a, s'), 0 if not specified
                    r = rewards.get((s, a, s2), 0)
                    # Bellman equation V(s) = sum over s, a, r of p(s'|s,a) * (r + yV(s'))
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                if v > best_value:
                    best_value = v
                    new_a = a

            # new_a now represents the best action in this state
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

        if is_policy_converged:
            break

    # once we're done, print the final policy and values
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
