from enum import Enum
import numpy as np
import random

# Constant states
class CellState(Enum):
    EMPTY = 0   # -> Forest
    FIRE = 1    # -> Burned
    WATER = 2   # Stays constant
    FLOOD = 3   # -> Empty
    FOREST = 4  # -> Overgrown Forest
    OVERGORWN_FOREST = 5  # Stays constant. Burns longer
    BURNED = 6  # -> Empty

# Time transitions
BURN_DURATION_MIN = 1
BURN_DURATION_MAX = 5

RUINED_DURATION_MIN = 50
RUINED_DURATION_MAX = 75

GROW_DURATION_MIN = 100
GROW_DURATION_MAX = 125

OVERGROW_DURATION_MIN = 500
OVERGROW_DURATION_MAX = 700

class WindDirection(Enum):
    NONE = 0
    N = 1
    S = 2
    W = 3
    E = 4
    NW = 5
    SW = 6
    NE = 7
    SE = 8

# Automaton class
class CellularAutomaton:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # Initialize empty grid
        self.initial_grid = self.grid.copy()
        self.wind_direction = WindDirection.NONE  # Default: no wind
        self.fire_spread_chance = 0.8  # Default 80% chance to spread fire
        self.timers = np.full((rows, cols), -1, dtype=int)  # -1 means no fire

    def initialize_from_map(self, map_array):
        """Initialize the automaton grid based on a given map."""
        self.grid = np.array(map_array)
        self.initial_grid = self.grid.copy()

    def update(self):
        """Update the automaton grid based on transition rules."""
        new_grid = self.grid.copy()

        for row in range(self.rows):
            for col in range(self.cols):
                state = self.grid[row, col]

                # Decrease cell timer
                if self.timers[row, col] > 0:
                    self.timers[row, col] -= 1
                elif self.timers[row, col] == 0:
                    # Tranistion cell's state if timer runs out
                    self.transition_cell(row, col, state, new_grid)

                # Spread fire with weighted probabilities
                if state == CellState.FIRE.value:
                    neighbors_with_weights = self.get_neighbors_with_wind_and_weights(row, col)
                    for (nr, nc), weight in neighbors_with_weights:
                        if self.grid[nr, nc] == CellState.FOREST.value:
                            if random.random() < self.fire_spread_chance * weight:
                                new_grid[nr, nc] = CellState.FIRE.value
                                self.init_cell(nr, nc, CellState.FIRE.value)

        self.grid = new_grid

    def init_cell(self, row, col, state):
        if state == CellState.EMPTY.value:
            self.timers[row, col] = random.randint(GROW_DURATION_MIN, GROW_DURATION_MAX)
        elif state == CellState.FIRE.value:
            self.timers[row, col] = random.randint(BURN_DURATION_MIN, BURN_DURATION_MAX)
        elif state == CellState.WATER.value:
            self.timers[row, col] = -1
        elif state == CellState.FLOOD.value:
            self.timers[row, col] = random.randint(RUINED_DURATION_MIN, RUINED_DURATION_MAX)
        elif state == CellState.FOREST.value:
            self.timers[row, col] = random.randint(OVERGROW_DURATION_MIN, OVERGROW_DURATION_MIN)
        elif state == CellState.OVERGORWN_FOREST.value:
            self.timers[row, col] = -1
        elif state == CellState.BURNED.value:
            self.timers[row, col] = random.randint(RUINED_DURATION_MIN, RUINED_DURATION_MAX)

    # Updates within old, not new grid
    def put_cell(self, row, col, value):
        self.grid[row, col] = value
        self.init_cell(row, col, value)

    def transition_cell(self, row, col, state, new_grid):
        new_state = state
        if state == CellState.EMPTY.value:
            new_state = CellState.FOREST.value
        elif state == CellState.FIRE.value:
            new_state = CellState.BURNED.value
        elif state == CellState.FLOOD.value:
            new_state = CellState.EMPTY.value
        elif state == CellState.FOREST.value:
            new_state = CellState.OVERGORWN_FOREST.value
        elif state == CellState.BURNED.value:
            new_state = CellState.EMPTY.value
        else:   # Unhandled cell type
            return

        # Update and initialize the cell
        new_grid[row, col] = new_state
        self.init_cell(row, col, new_state)

    def reset(self):
        self.grid = self.initial_grid.copy()
        self.timers = np.full((self.rows, self.cols), -1, dtype=int)  # Reset timers

    def set_wind(self, direction: WindDirection, intensity: int = 1):
        """Set the wind direction and adjust fire spread chance."""
        self.wind_direction = direction
        self.fire_spread_chance = 0.5 + (intensity * 0.05)  # Increase chance with intensity
        self.fire_spread_chance = min(self.fire_spread_chance, 1.0)  # Clamp to max 100%

    def get_neighbors(self, row, col):
        """Get valid neighbors for a given cell."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def get_neighbors_with_wind_and_weights(self, row, col):
        """Get neighbors with weights based on wind direction."""
        # Direction mapping with weights
        direction_weights = {
            WindDirection.N: [((-1, 0), 1.0), ((-1, -1), 0.7), ((-1, 1), 0.7)],
            WindDirection.S: [((1, 0), 1.0), ((1, -1), 0.7), ((1, 1), 0.7)],
            WindDirection.W: [((0, -1), 1.0), ((-1, -1), 0.7), ((1, -1), 0.7)],
            WindDirection.E: [((0, 1), 1.0), ((-1, 1), 0.7), ((1, 1), 0.7)],
            WindDirection.NW: [((-1, -1), 1.0), ((-1, 0), 0.8), ((0, -1), 0.8)],
            WindDirection.NE: [((-1, 1), 1.0), ((-1, 0), 0.8), ((0, 1), 0.8)],
            WindDirection.SW: [((1, -1), 1.0), ((1, 0), 0.8), ((0, -1), 0.8)],
            WindDirection.SE: [((1, 1), 1.0), ((1, 0), 0.8), ((0, 1), 0.8)],
            WindDirection.NONE: [((-1, 0), 0.3), ((1, 0), 0.3), ((0, -1), 0.3), ((0, 1), 0.3),
                                 ((-1, -1), 0.2), ((-1, 1), 0.2), ((1, -1), 0.2), ((1, 1), 0.2)],
        }

        # Get neighbors based on wind direction
        wind_neighbors = direction_weights.get(self.wind_direction, [])
        neighbors_with_weights = []

        for (dr, dc), weight in wind_neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors_with_weights.append(((nr, nc), weight))

        return neighbors_with_weights



