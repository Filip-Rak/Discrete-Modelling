from enum import Enum
import numpy as np
import random

# Constant states
class CellState(Enum):
    EMPTY = 0
    FIRE = 1
    WATER = 2
    FOREST = 3
    BURNED = 4

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

                if state == CellState.FIRE.value:
                    # Spread fire with weighted probabilities
                    neighbors_with_weights = self.get_neighbors_with_wind_and_weights(row, col)
                    for (nr, nc), weight in neighbors_with_weights:
                        if self.grid[nr, nc] == CellState.FOREST.value:
                            if random.random() < self.fire_spread_chance * weight:
                                new_grid[nr, nc] = CellState.FIRE.value

        self.grid = new_grid

    def reset(self):
        self.grid = self.initial_grid.copy()

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



