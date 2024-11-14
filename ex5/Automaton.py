from enum import Enum
import numpy as np

# Define cell states
class CellState(Enum):
    EMPTY = 0
    FIRE = 1
    WATER = 2
    FOREST = 3
    BURNED = 4

# Automaton class
class CellularAutomaton:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # Initialize empty grid
        self.initial_grid = self.grid.copy()

    def initialize_from_map(self, map_array):
        """Initialize the automaton grid based on a given map."""
        self.grid = np.array(map_array)
        self.initial_grid = self.grid.copy()

    def update(self):
        """Update the automaton grid based on transition rules."""
        new_grid = self.grid.copy()
        # Add your transition rules here
        # Example: spreading fire
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == CellState.FIRE.value:
                    neighbors = self.get_neighbors(row, col)
                    for nr, nc in neighbors:
                        if self.grid[nr, nc] == CellState.FOREST.value:
                            new_grid[nr, nc] = CellState.FIRE.value
        self.grid = new_grid

    def reset(self):
        self.grid = self.initial_grid.copy()

    def get_neighbors(self, row, col):
        """Get valid neighbors for a given cell."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors
