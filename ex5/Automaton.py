from enum import Enum
import numpy as numpy
import random

# Constants
# --------------------
# States of the cell
class CellState(Enum):
    EMPTY = 0   # -> Forest
    FIRE = 1    # -> Burned
    WATER = 2   # Stays constant
    FLOOD = 3   # -> Empty
    FOREST = 4  # -> Overgrown Forest
    OVERGROWN_FOREST = 5  # Stays constant. Burns longer
    BURNED = 6  # -> Empty

# Time transitions
BURN_DURATION_MIN = 1
BURN_DURATION_MAX = 5
OVERGROWN_BURN_MULTIPLIER = 6

RUINED_DURATION_MIN = 50
RUINED_DURATION_MAX = 75

GROW_DURATION_MIN = 150
GROW_DURATION_MAX = 175

OVERGROW_DURATION_MIN = 500
OVERGROW_DURATION_MAX = 700

# Directions of the wind
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

# Chance of fire spreading based on wind direction
WIND_DIRECTION_WEIGHT = 0.9
PARALLER_DIRECTION_WEIGHT = 0.7
OPPOSITE_DIRECTION_WEIGHT = 0.05
NEUTRAL_DIRECTION_WEIGHT = 0.1

class CellularAutomaton:
    # Constructor
    # ----------------
    def __init__(self, rows: int, cols: int):
        # Grid setup
        self.rows = rows
        self.cols = cols
        self.grid = numpy.zeros((rows, cols), dtype=int)  # Initialize empty grid
        self._initial_grid = self.grid.copy()

        # Update modifiers
        self.wind_direction = WindDirection.NONE  # Default: no wind
        self._fire_spread_chance = 0.5  # Default: 80% chance to spread fire
        self._timers = numpy.full((rows, cols), -1, dtype=int)  # -Default: -1 means no time transition

    # Public Methods
    # ----------------
    def initialize_from_map(self, map_array: numpy.ndarray):
        """Initialize the automaton grid based on a given map."""
        # Copy the grid to an array
        self.grid = numpy.array(map_array)

        # Initialize all the cells
        for row in range(self.rows):
            for col in range(self.cols):
                state = self.grid[row, col]
                self._init_dell(row, col, state)

        # Make a copy for restarts
        self._initial_grid = self.grid.copy()

    def update(self):
        """Update the automaton grid based on rules."""

        # Work on grid's copy
        new_grid = self.grid.copy()

        # Iterates through grid matrix
        for row in range(self.rows):
            for col in range(self.cols):
                state = self.grid[row, col]

                # Decrease cell timer
                if self._timers[row, col] > 0:
                    self._timers[row, col] -= 1
                elif self._timers[row, col] == 0:
                    # Tranistion cell's state if timer runs out
                    self._transition_cell(row, col, state, new_grid)

                # Spread fire with weighted probabilities
                if state == CellState.FIRE.value:
                    neighbors_with_weights = self._get_neighbors_with_wind_and_weights(row, col)
                    for (nr, nc), weight in neighbors_with_weights:
                        if self.grid[nr, nc] == CellState.FOREST.value or self.grid[nr, nc] == CellState.OVERGROWN_FOREST.value:
                            if random.random() < self._fire_spread_chance * weight:
                                new_grid[nr, nc] = CellState.FIRE.value
                                self._init_dell(nr, nc, CellState.FIRE.value)

                # Spread water to neighboring cells
                elif state == CellState.WATER.value or state == CellState.FLOOD.value:
                    neighbors = self._get_neighbors(row, col)
                    for nr, nc in neighbors:
                        if self.grid[nr, nc] in [CellState.FIRE.value, CellState.BURNED.value]:
                            new_grid[nr, nc] = CellState.FLOOD.value
                            self._init_dell(nr, nc, CellState.FLOOD.value)

        # Update the grid
        self.grid = new_grid

    def reset(self):
        """Return the state of the grid to initial form"""
        self.grid = self._initial_grid.copy()
        self._timers = numpy.full((self.rows, self.cols), -1, dtype=int)  # Reset timers

        # Reinitialize the grid
        # Initialize all the cells
        for row in range(self.rows):
            for col in range(self.cols):
                state = self.grid[row, col]
                self._init_dell(row, col, state)

    def put_cell(self, row: int, col: int, state: CellState):
        """Puts and initializes cell, with a new given state, directly into the grid"""
        self.grid[row, col] = state
        self._init_dell(row, col, state)

    def set_wind(self, direction: WindDirection):
        """Set the wind direction adjusting fire spread chance."""
        self.wind_direction = direction

    # Private Methods
    # ----------------
    def _init_dell(self, row: int, col: int, state: CellState):
        """Initializes cell timer within the matrix"""
        if state == CellState.EMPTY.value:
            self._timers[row, col] = random.randint(GROW_DURATION_MIN, GROW_DURATION_MAX)
        elif state == CellState.FIRE.value:
            if self.grid[row, col] == CellState.OVERGROWN_FOREST.value:
                self._timers[row, col] = random.randint(BURN_DURATION_MIN * OVERGROWN_BURN_MULTIPLIER, BURN_DURATION_MAX * OVERGROWN_BURN_MULTIPLIER)
            else:
                self._timers[row, col] = random.randint(BURN_DURATION_MIN, BURN_DURATION_MAX)
        elif state == CellState.WATER.value:
            self._timers[row, col] = -1
        elif state == CellState.FLOOD.value:
            self._timers[row, col] = random.randint(RUINED_DURATION_MIN, RUINED_DURATION_MAX)
        elif state == CellState.FOREST.value:
            self._timers[row, col] = random.randint(OVERGROW_DURATION_MIN, OVERGROW_DURATION_MAX)
        elif state == CellState.OVERGROWN_FOREST.value:
            self._timers[row, col] = -1
        elif state == CellState.BURNED.value:
            self._timers[row, col] = random.randint(RUINED_DURATION_MIN, RUINED_DURATION_MAX)

    def _transition_cell(self, row: int, col: int, state: CellState, grid: numpy.ndarray):
        """Update the cell to it's next state"""
        new_state = state
        if state == CellState.EMPTY.value:
            new_state = CellState.FOREST.value
        elif state == CellState.FIRE.value:
            new_state = CellState.BURNED.value
        elif state == CellState.FLOOD.value:
            new_state = CellState.EMPTY.value
        elif state == CellState.FOREST.value:
            new_state = CellState.OVERGROWN_FOREST.value
        elif state == CellState.BURNED.value:
            new_state = CellState.EMPTY.value
        else:   # Unhandled cell type
            return

        # Update and initialize the cell
        grid[row, col] = new_state
        self._init_dell(row, col, new_state)

    def _get_neighbors(self, row: int, col: int) -> list:
        """Get valid neighbors for a given cell, including diagonals."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def _get_neighbors_with_wind_and_weights(self, row, col) -> list:
        """Get neighbors with weights based on wind direction."""

        # Weights with orienation to wind
        wfd = WIND_DIRECTION_WEIGHT
        wpd = PARALLER_DIRECTION_WEIGHT
        wod = OPPOSITE_DIRECTION_WEIGHT
        nwd = NEUTRAL_DIRECTION_WEIGHT

        # Direction mapping with weights
        direction_weights = {
            WindDirection.N: [((-1, 0), wfd), ((-1, -1), wpd), ((-1, 1), wpd),
                              ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.S: [((1, 0), wfd), ((1, -1), wpd), ((1, 1), wpd),
                              ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.W: [((0, -1), wfd), ((-1, -1), wpd), ((1, -1), wpd),
                              ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.E: [((0, 1), wfd), ((-1, 1), wpd), ((1, 1), wpd),
                              ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.NW: [((-1, -1), wfd), ((-1, 0), wpd), ((0, -1), wpd),
                               ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.NE: [((-1, 1), wfd), ((-1, 0), wpd), ((0, 1), wpd),
                               ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.SW: [((1, -1), wfd), ((1, 0), wpd), ((0, -1), wpd),
                               ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.SE: [((1, 1), wfd), ((1, 0), wpd), ((0, 1), wpd),
                               ((-1, -1), wod), ((-1, 1), wod), ((1, -1), wod), ((1, 1), wod)],
            WindDirection.NONE: [((-1, 0), nwd), ((1, 0), nwd), ((0, -1), nwd), ((0, 1), nwd),
                                 ((-1, -1), nwd), ((-1, 1), nwd), ((1, -1), nwd), ((1, 1), nwd)],
        }

        # Get neighbors based on wind direction
        wind_neighbors = direction_weights.get(self.wind_direction, [])
        neighbors_with_weights = []

        for (dr, dc), weight in wind_neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors_with_weights.append(((nr, nc), weight))

        return neighbors_with_weights
