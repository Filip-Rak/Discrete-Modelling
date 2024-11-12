from PIL import Image, ImageDraw
from enum import Enum
import numpy

# Constants
# --------------------
class Pattern(Enum):
    GLIDER = 1
    OSCILLATOR = 2
    STILL = 3
    RANDOM = 4

class BoundaryCondition(Enum):
    PERIODIC = 1    # Wrap-around / torus
    REFLECTIVE = 2  # Mirroring

# Cell
CELL_ACTIVE = 1
CELL_INACTIVE = 0

# State
MIN_WIDTH = 1
MIN_HEIGHT = 1

# Helper Functions
# --------------------
def handle_boundry(x, y, rows, cols, boundary_condition: BoundaryCondition):
    if boundary_condition == BoundaryCondition.PERIODIC:
        # Wrap-around / torus
        x_wrapped = x % rows
        y_wrapped = y % cols
        return x_wrapped, y_wrapped

    elif boundary_condition == BoundaryCondition.REFLECTIVE:
        # Mirroring
        x_reflect = max(0, min(rows - 1, x))
        y_reflect = max(0, min(cols - 1, y))
        return x_reflect, y_reflect

    else:
        raise ValueError("Invalid boundary condition")

def get_pattern(pattern: Pattern, rand_x, rand_y) -> numpy.ndarray:
    if pattern == Pattern.GLIDER:
        return numpy.array([
            [CELL_INACTIVE, CELL_ACTIVE, CELL_INACTIVE],
            [CELL_INACTIVE, CELL_INACTIVE, CELL_ACTIVE],
            [CELL_ACTIVE, CELL_ACTIVE, CELL_ACTIVE]
        ])
    elif pattern == Pattern.OSCILLATOR:
        return numpy.array([
            [CELL_INACTIVE, CELL_ACTIVE, CELL_INACTIVE],
            [CELL_INACTIVE, CELL_ACTIVE, CELL_INACTIVE],
            [CELL_INACTIVE, CELL_ACTIVE, CELL_INACTIVE]
        ])
    elif pattern == Pattern.STILL:
        return numpy.array([
            [CELL_ACTIVE, CELL_ACTIVE],
            [CELL_ACTIVE, CELL_ACTIVE]
        ])
    elif pattern == Pattern.RANDOM:
        return numpy.random.choice([CELL_INACTIVE, CELL_ACTIVE], size=(rand_x, rand_y))
    else:
        raise ValueError("Invalid pattern")

# Exportable Functions
# --------------------
def get_initial_state(width : int, height : int, pattern : Pattern, rand_x = 3, rand_y = 3) -> numpy.ndarray:
    # Debug output
    print("Getting initial state...")

    # Validate dimensions
    width = max(width, MIN_WIDTH)
    height = max(height, MIN_HEIGHT)
    rand_x = min(rand_x, width)
    rand_y = min(rand_y, height)

    # Initialize an empty matrix
    output_matrix = numpy.zeros((height, width), dtype=int)

    # Get the pattern matrix
    pattern_matrix = get_pattern(pattern, rand_x, rand_y)
    pattern_height, pattern_width = pattern_matrix.shape

    # Calculate the top-left position to center the pattern
    start_x = (width - pattern_width) // 2
    start_y = (height - pattern_height) // 2

    # Place the pattern in the center of the output matrix
    output_matrix[start_y : start_y + pattern_height, start_x : start_x + pattern_width] = pattern_matrix

    return output_matrix

def apply_rules(seed_matrix: numpy.ndarray, iterations: int, boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC) -> list:
    # Debug output
    print("Applying rules for %d iterations [%%]: " % iterations, end="")

    rows, cols = seed_matrix.shape
    current_state = seed_matrix.copy()

    # List all active cells
    active_cells = set((i, j) for i in range(rows) for j in range(cols) if current_state[i, j] == CELL_ACTIVE)
    states = [current_state.copy()]

    # Main iteration loop
    for i in range(iterations):
        # Debug output co 10%
        if i % (iterations // 10) == 0:
            percent = (i / iterations) * 100
            print("|%d| " % percent, end="")

        new_state = numpy.zeros((rows, cols), dtype=int)
        cells_to_check = set()

        # Add active cells and their neighbors to list of cells to check
        for (x, y) in active_cells:
            cells_to_check.add((x, y))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i, j) != (0, 0):
                        # Zastosowanie warunku brzegowego przy dodawaniu sąsiadów
                        nx, ny = handle_boundry(x + i, y + j, rows, cols, boundary_condition)
                        cells_to_check.add((nx, ny))

        new_active_cells = set()

        # Only compute cells to check
        for (x, y) in cells_to_check:
            # Get the number of neighbors, uwzględniając warunek brzegowy
            live_neighbors = sum(
                current_state[handle_boundry(x + i, y + j, rows, cols, boundary_condition)]
                for i in [-1, 0, 1]
                for j in [-1, 0, 1]
                if (i, j) != (0, 0)
            )

            # Apply rules
            if current_state[x, y] == CELL_ACTIVE:
                # Activate cell if 2 or 3 neighbors alive
                if live_neighbors == 2 or live_neighbors == 3:
                    new_state[x, y] = CELL_ACTIVE
                    new_active_cells.add((x, y))
            else:
                # Activate cell if 3 neighbors are alive
                if live_neighbors == 3:
                    new_state[x, y] = CELL_ACTIVE
                    new_active_cells.add((x, y))

        # Update the game state
        current_state = new_state
        active_cells = new_active_cells
        states.append(current_state.copy())

    # Debug output
    print(end="\n")

    return states

def save_as_gif(states: list, filename: str, cell_size: int = 10, duration=20, skip_frames=1):
    # Colors as RGB
    color_active = [57, 255, 20]
    color_inactive = [50, 50, 50]

    # Debug output
    print("Converting states to images for %d states[%%]: " % len(states), end="")

    frames = []
    i = 0
    for state in states:
        if i % skip_frames == 0:
            rows, cols = state.shape

            # Make RGB array out of the state array
            img_array = numpy.zeros((rows, cols, 3), dtype=numpy.uint8)
            img_array[state == CELL_ACTIVE] = color_active
            img_array[state != CELL_ACTIVE] = color_inactive

            # Conversion to PIL image
            img = Image.fromarray(img_array, mode="RGB")
            img = img.resize((cols * cell_size, rows * cell_size), resample=Image.NEAREST)

            frames.append(img)

        i += 1

        # Debug output
        if i % (len(states) // 11) == 0:
            percent = (i / len(states)) * 100
            print("|%d| " % percent, end="")

    # Saving images as a gif
    print("\nSaving as a gif...")
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2
    )