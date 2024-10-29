from typing import Tuple
from enum import Enum
import numpy

# --------------------
# Constants
RULE_MAX = 255
RULE_MIN = 0
STATE_MIN = 0
STATE_MAX = 1
RETURNED_VALUE = -1

class BorderHandling(Enum):
    ZERO_PADDING = 1    # Fills with zeros
    ONE_PADDING = 2     # Fills with ones
    REPLICATE_EDGE = 3  # Repeats edge value
    MIRROR = 4          # Mirrors in respect to the border
    PERIODIC = 5        # Wraps around

# Helper Functions
# --------------------
def handle_border(matrix : numpy.ndarray, index_x : int, index_y : int, handler : BorderHandling) -> Tuple[int, int]:
    # Return given cords if inside the matrix
    rows, cols = matrix.shape
    if 0 <= index_x < rows and 0 <= index_y < cols:
        return index_x, index_y

    # Otherwise, use boundary conditions
    # Fills with zeros
    if handler == BorderHandling.ZERO_PADDING:
        return RETURNED_VALUE, 0

    # Fills with ones
    elif handler == BorderHandling.ONE_PADDING:
        return RETURNED_VALUE, 1

    # Repeats edge value
    elif handler == BorderHandling.REPLICATE_EDGE:
        index_x = min(max(index_x, 0), rows - 1)
        index_y = min(max(index_y, 0), cols - 1)
        return index_x, index_y

    # Mirrors in respect to the border
    elif handler == BorderHandling.MIRROR:
        index_x = abs(index_x) if index_x < 0 else (2 * rows - 2 - index_x) if index_x >= rows else index_x
        index_y = abs(index_y) if index_y < 0 else (2 * cols - 2 - index_y) if index_y >= cols else index_y
        return index_x, index_y

    # Wraps around
    elif handler == BorderHandling.PERIODIC:
        index_x = index_x % rows
        index_y = index_y % cols
        return index_x, index_y

    return index_x, index_y

def get_rule_table(rule: int) -> numpy.ndarray:
    # Convert the rule number to 8-bit binary
    binary_string = f"{rule:08b}"
    binary_string = binary_string[::-1]   # Invert order to align with 0 to 7 indecies

    # Return array with results for each neighbour configuration
    return numpy.array([int(bit) for bit in binary_string], dtype=int)

# Exportable Functions
# --------------------
def generate_seed(size : int):
    return numpy.random.randint(STATE_MIN, STATE_MAX + 1, size)

def apply_rule(rule : int, seed : numpy.ndarray, rows : int, iterations : int,
    handler : BorderHandling = BorderHandling.PERIODIC) -> numpy.ndarray:
    # Create result matrix
    columns = seed.shape[0]
    result_matrix = numpy.zeros((rows, columns), dtype=int)
    result_matrix[0] = seed # Save the seed as first row

    # Clamp the rule and prepare the table
    rule = numpy.clip(rule, RULE_MIN, RULE_MAX)
    rule_table = get_rule_table(rule)

    # Limit iterations to size
    iterations = min(iterations, rows)

    # Apply the rule to each cell
    for i in range(1, iterations):
        for j in range(0, columns):
            # Get neighbours
            center = result_matrix[i - 1][j]

            # Left neighbour
            li = handle_border(result_matrix, i - 1, j - 1, handler)
            if li[0] == RETURNED_VALUE:
                left = li[1]
            else:
                left = result_matrix[li[0]][li[1]]

            # Right neighbour
            ri = handle_border(result_matrix, i - 1, j + 1, handler)
            if ri[0] == RETURNED_VALUE:
                right = ri[1]
            else:
                right = result_matrix[ri[0]][ri[1]]

            # Get index of this neighborhood type
            neighborhood = (left << 2) | (center << 1) | right

            # Apply the rule
            result_matrix[i, j] = rule_table[neighborhood]

    return result_matrix



