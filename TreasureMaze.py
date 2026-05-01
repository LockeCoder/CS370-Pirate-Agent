# This class represents the environment, which includes a maze object defined as a matrix.

import numpy as np


visited_mark = 0.8  # The visited cells are marked by an 80% gray shade.
pirate_mark = 0.5   # The current cell where the pirate is located is marked by a 50% gray shade.

# The agent can move in one of four directions.
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class TreasureMaze(object):
    """
    Gridworld where 1.0 = free cell and 0.0 = wall.

    Pirate starts at a free cell and tries to reach bottom-right treasure.

    Rewards:
        +1.0 on reaching treasure
        -1.0 beyond min_reward when blocked, which ends the game
        -0.75 for invalid action with no move
        -0.25 for revisiting a cell
        -0.04 for a valid step to a new cell

    Game status strings:
        running
        win
        lose
    """

    def __init__(self, maze, pirate=(0, 0)):
        self._maze = np.array(maze, dtype=float)

        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)

        # Build list of free cells, excluding the target so all starts test toward the target.
        self.free_cells = [
            (r, c)
            for r in range(nrows)
            for c in range(ncols)
            if self._maze[r, c] == 1.0
        ]

        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")

        if self.target in self.free_cells:
            self.free_cells.remove(self.target)

        if pirate not in self.free_cells:
            raise Exception("Invalid Pirate Location: must sit on a free cell")

        # Reasonable floor for losing the game: half the number of cells.
        self.min_reward = -0.5 * self._maze.size

        self.reset(pirate)

    @property
    def maze(self):
        return self._maze

    @property
    def rows(self):
        return self._maze.shape[0]

    @property
    def cols(self):
        return self._maze.shape[1]

    def reset(self, pirate):
        """Reset the pirate's position."""
        self.state = (pirate[0], pirate[1], "start")
        self.visited = set()
        self.total_reward = 0.0

        return self.observe()

    def update_state(self, action):
        """Update internal state based on an action."""
        pirate_row, pirate_col, mode = self.state

        nrow = pirate_row
        ncol = pirate_col
        nmode = mode

        # Mark visited if this cell is free.
        if self.maze[pirate_row, pirate_col] > 0.0:
            self.visited.add((pirate_row, pirate_col))

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = "blocked"
        elif action in valid_actions:
            nmode = "valid"

            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:
            # Invalid action, no move.
            nmode = "invalid"

        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        """Return a reward based on the agent movement guidelines."""
        pirate_row, pirate_col, mode = self.state
        nrows, ncols = self.maze.shape

        if pirate_row == nrows - 1 and pirate_col == ncols - 1:
            return 1.0

        if mode == "blocked":
            return self.min_reward - 1.0

        if (pirate_row, pirate_col) in self.visited:
            return -0.25

        if mode == "invalid":
            return -0.75

        if mode == "valid" or mode == "start":
            return -0.04

        return -0.04

    def act(self, action):
        """Track state and total reward based on the agent action."""
        self.update_state(action)

        reward = self.get_reward()
        self.total_reward += reward

        status = self.game_status()
        envstate = self.observe()

        return envstate, reward, status

    def observe(self):
        """Return the current environment state as a 1 x rows*cols vector."""
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))

        return envstate

    def draw_env(self):
        """Return a render-only array for visualization or neural network input."""
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape

        # Clear all visual marks.
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0

        # Draw the pirate.
        row, col, _ = self.state
        canvas[row, col] = pirate_mark

        return canvas

    def game_status(self):
        """Return the current game status."""
        if self.total_reward < self.min_reward:
            return "lose"

        pirate_row, pirate_col, _ = self.state
        nrows, ncols = self.maze.shape

        if pirate_row == nrows - 1 and pirate_col == ncols - 1:
            return "win"

        return "running"

    def valid_actions(self, cell=None):
        """Return the set of valid actions from the current cell or a provided cell."""
        if cell is None:
            row, col, _ = self.state
        else:
            row, col = cell

        actions = [LEFT, UP, RIGHT, DOWN]
        nrows, ncols = self.maze.shape

        # Borders.
        if row == 0:
            actions.remove(UP)
        elif row == nrows - 1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == ncols - 1:
            actions.remove(RIGHT)

        # Walls.
        if row > 0 and self.maze[row - 1, col] == 0.0 and UP in actions:
            actions.remove(UP)

        if row < nrows - 1 and self.maze[row + 1, col] == 0.0 and DOWN in actions:
            actions.remove(DOWN)

        if col > 0 and self.maze[row, col - 1] == 0.0 and LEFT in actions:
            actions.remove(LEFT)

        if col < ncols - 1 and self.maze[row, col + 1] == 0.0 and RIGHT in actions:
            actions.remove(RIGHT)

        return actions
