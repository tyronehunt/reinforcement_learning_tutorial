from __future__ import print_function, division
from builtins import range
import numpy as np

# You can go up, down, left or right.
ACTION_SPACE = ('U', 'D', 'L', 'R')


# Set up the environment
class Grid:
    def __init__(self, rows, cols, start):
        """
    Create the environment (which is a grid of num rows/cols and a starting position)
    """
        self.rows = rows
        self.cols = cols
        # Note, rows count down, columns count left to right from (0,0)
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        """
    rewards: dict: (i, j): r (row, col): reward
    actions: dict: (i, j): A (row, col): list of possible actions
    """
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        """
    This is a "cheat" in the sense that it allows you to override to a state of your choice.
    """
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        """ Returns what current state is as a tuple """
        return (self.i, self.j)

    def is_terminal(self, s):
        """ Check if a state, s, is terminal - i.e. cannot move from there"""
        return s not in self.actions

    def get_next_state(self, s, a):
        """ Note, this only makes sense in this environment, because deterministic.
    It is a hypothetical for information only function, as doesn't perform the action in the environment.
    """
        # this answers: where would I end up if I perform action 'a' in state 's'?
        i, j = s[0], s[1]

        # if this action moves you somewhere else, then it will be in this dictionary
        if a in self.actions[(i, j)]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'R':
                j += 1
            elif a == 'L':
                j -= 1
        return i, j

    def move(self, action):
        """ Move to the next state and fetch the reward from that state."""
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        """ Reverses the move() function. Assert at the end checks our state is not defined."""
        # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert (self.current_state() in self.all_states())

    def game_over(self):
        """returns true if game is over, else false
    Game is over is true if we are in a state where no actions are possible
    """
        return (self.i, self.j) not in self.actions

    def all_states(self):
        """ Some states are not in actions dictionary, such as terminal states.
    Some states are not in rewards dictionary. Hence need untion to get all states.
    """
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    """ Defines a grid that describes reward for arriving in each state, and possible actions next.
  x means you can't go there.
  s means start position.
  number is reward at that state.
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  """
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves
    # so we will penalize every move
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g


class WindyGrid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions, probs):
        # Note now probs also in set function
        # rewards: dict of: (i, j): r (row, col): reward
        # actions: dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        """ This is new from regular grid class. We get the list of possible next states and their probabilities.
    We make a random choice, with associated probabilities with np.random.choice.
    We return the reward for this new state.
    """
        s = (self.i, self.j)
        a = action

        next_state_probs = self.probs[(s, a)]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        s2 = np.random.choice(next_states, p=next_probs)

        # update the current state
        self.i, self.j = s2

        # return a reward (if any)
        return self.rewards.get(s2, 0)

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())


def windy_grid():
    """ Initiate the environment with the actions, rewards and state transitions."""
    g = WindyGrid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }
    g.set(rewards, actions, probs)
    return g


def windy_grid_penalized(step_cost=-0.1):
    g = WindyGrid(3, 4, (2, 0))
    rewards = {
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
        (0, 3): 1,
        (1, 3): -1
    }
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }
    g.set(rewards, actions, probs)
    return g


def grid_5x5(step_cost=-0.1):
    g = Grid(5, 5, (4, 0))
    rewards = {(0, 4): 1, (1, 4): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'R'),
        (0, 3): ('L', 'D', 'R'),
        (1, 0): ('U', 'D', 'R'),
        (1, 1): ('U', 'D', 'L'),
        (1, 3): ('U', 'D', 'R'),
        (2, 0): ('U', 'D', 'R'),
        (2, 1): ('U', 'L', 'R'),
        (2, 2): ('L', 'R', 'D'),
        (2, 3): ('L', 'R', 'U'),
        (2, 4): ('L', 'U', 'D'),
        (3, 0): ('U', 'D'),
        (3, 2): ('U', 'D'),
        (3, 4): ('U', 'D'),
        (4, 0): ('U', 'R'),
        (4, 1): ('L', 'R'),
        (4, 2): ('L', 'R', 'U'),
        (4, 3): ('L', 'R'),
        (4, 4): ('L', 'U'),
    }
    g.set(rewards, actions)

    # non-terminal states
    visitable_states = actions.keys()
    for s in visitable_states:
        g.rewards[s] = step_cost

    return g
