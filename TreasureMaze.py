# This class represents the environment, which includes a maze object defined as a matrix.

import numpy as np

visited_mark = 0.8  # The visited cells are marked by an 80% gray shade.
pirate_mark  = 0.5  # The current cell where the pirate is located is marked by a 50% gray shade.

# The agent can move in one of four directions.
LEFT  = 0
UP    = 1
RIGHT = 2
DOWN  = 3

class TreasureMaze(object):
    """
    Gridworld where 1.0 = free cell and 0.0 = wall.
    Pirate starts at a free cell and tries to reach bottom-right (treasure).
    Rewards:
      +1.0 on reaching treasure
      -1.0 beyond min_reward when blocked (game over)
      -0.75 for invalid action (no move)
      -0.25 for revisiting a cell
      -0.04 for a valid step to a new cell
    Game status strings are: 'running', 'win', 'lose'
    """

    def __init__(self, maze, pirate=(0, 0)):
        self._maze = np.array(maze, dtype=float)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)       
        # build list of free cells (exclude target to test all starts to target)
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if self.target in self.free_cells:
            self.free_cells.remove(self.target)
        if pirate not in self.free_cells:
            raise Exception("Invalid Pirate Location: must sit on a free cell")
        # reasonable floor for losing the game: half the number of cells
        self.min_reward = -0.5 * self._maze.size
        self.reset(pirate)

    # Public helpers 
    @property
    def maze(self):
        return self._maze

    @property
    def rows(self):
        return self._maze.shape[0]

    @property
    def cols(self):
        return self._maze.shape[1]

    # This method resets the pirate's position.
    def reset(self, pirate):
        self.state = (pirate[0], pirate[1], 'start')
        self.visited = set()
        self.total_reward = 0.0
        return self.observe()

    # Update internal state based on an action (no reward bookkeeping here).
    def update_state(self, action):
        nrows, ncols = self.maze.shape
        # unpack current state
        pirate_row, pirate_col, mode = self.state
        nrow, ncol = pirate_row, pirate_col
        nmode = mode

        # mark visited if this cell is free
        if self.maze[pirate_row, pirate_col] > 0.0:
            self.visited.add((pirate_row, pirate_col))

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:
            # invalid action, no move
            nmode = 'invalid'

        # set new state
        self.state = (nrow, ncol, nmode)

    # This method returns a reward based on the agent movement guidelines.
    def get_reward(self):
        pirate_row, pirate_col, mode = self.state
        nrows, ncols = self.maze.shape
        if pirate_row == nrows - 1 and pirate_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1.0
        if (pirate_row, pirate_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid' or mode == 'start':
            return -0.04
        return -0.04

    # This method keeps track of the state and total reward based on agent action.
    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    # This method returns the current environment state as a 1 x (rows*cols) vector.
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    # Render-only array (no side effects) for visualization or NN input.
    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the pirate
        row, col, _ = self.state
        canvas[row, col] = pirate_mark
        return canvas

    # This method returns the game status.
    def game_status(self):
        # If the agent’s total reward goes below the minimum reward, the game is over.
        if self.total_reward < self.min_reward:
            return 'lose'
        pirate_row, pirate_col, _ = self.state
        nrows, ncols = self.maze.shape
        # If the agent reaches the treasure cell, the game is won.
        if pirate_row == nrows - 1 and pirate_col == ncols - 1:
            return 'win'
        # Game is not complete yet
        return 'running'

    # This method returns the set of valid actions starting from the current cell (or a given cell).
    def valid_actions(self, cell=None):
        if cell is None:
            row, col, _ = self.state
        else:
            row, col = cell
        actions = [LEFT, UP, RIGHT, DOWN]
        nrows, ncols = self.maze.shape
        # borders
        if row == 0:
            actions.remove(UP)
        elif row == nrows - 1:
            actions.remove(DOWN)
        if col == 0:
            actions.remove(LEFT)
        elif col == ncols - 1:
            actions.remove(RIGHT)
        # walls
        if row > 0 and self.maze[row - 1, col] == 0.0 and UP in actions:
            actions.remove(UP)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0 and DOWN in actions:
            actions.remove(DOWN)
        if col > 0 and self.maze[row, col - 1] == 0.0 and LEFT in actions:
            actions.remove(LEFT)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0 and RIGHT in actions:
            actions.remove(RIGHT)
        return actions