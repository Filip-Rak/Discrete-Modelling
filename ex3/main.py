from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from ElementaryCa import *
from typing import Tuple
import numpy

# --------------------
# Constants

# Automaton
DEFAULT_WIDTH = 11
DEFAULT_HEIGHT = 20
DEFAULT_ITERATIONS = 20
DEFAULT_RULE = 150
DEFAULT_STATE = 0
DEFAULT_SEED = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

# Paths
OUTPUT_PATH = "Output/"

# --------------------
# Main Functions
def main():
    # Gather input
    width, height, iterations, rule = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ITERATIONS, DEFAULT_RULE
    seed_array = generate_seed(width)

    width, height, iterations, rule = gather_basic_input()

    # Get or generate seed
    seed_array = get_seed_input(width)
    if seed_array.shape[0] != width:    # Ensure the width is consistent
        width = seed_array.shape[0]

    # Apply the rule
    result_matrix = apply_rule(rule, seed_array, height, iterations, BorderHandling.ZERO_PADDING)
    print(result_matrix)

    # Make a graph
    visualize_matrix(result_matrix)

    # Save the matrix as CSV
    save_matrix(result_matrix, OUTPUT_PATH + "result_matrix.csv")

def gather_basic_input() -> Tuple[int, int, int, int]:
    # Gather inputs
    width = make_int(input("Width: "), DEFAULT_WIDTH)
    height = make_int(input("Height: "), DEFAULT_HEIGHT)
    iterations = make_int(input("Iterations: "), DEFAULT_ITERATIONS)
    rule = make_int(input("Rule: "), DEFAULT_RULE)

    # Clamp the rule
    rule = numpy.clip(rule, RULE_MIN, RULE_MAX)

    # Print result
    print(f"Basic Params: Width: {width}, Height: {height}, Iterations: {iterations}, Rule: {rule}")

    # Return inputs
    return width, height, iterations, rule

def get_seed_input(width : int):
    # Get input
    string_input = input("Seed [1, 0, 1]. Leave empty for random: ")

    # Check for special case
    if string_input == "":
        return generate_seed(width)

    # Convert to a number
    try:
        seed = list(map(int, string_input.split(", ")))
        for i in range(0, len(seed)):
            if seed[i] != 0 and seed[i] != 1:
                seed[i] = DEFAULT_STATE

        return numpy.array(seed)

    except ValueError:  # If failed, use default
        return DEFAULT_SEED

def save_matrix(matrix, filename, iterations = -1):
    # Clip the matrix if allowed
    matrix_to_save = matrix[:iterations].copy() if iterations > 0 else matrix

    # Save the matrix to file
    numpy.savetxt(OUTPUT_PATH + "result_matrix.csv", matrix_to_save, delimiter=",", fmt="%d")

# --------------------
# Helper Functions
def make_int(input, default : int) -> int:
    try:
        input = int(input)
        if input != 0:
            return abs(input) # Return if a non-zero number
    except ValueError:
        try:
            ascii = ord(str(input)[0]) # Convert NAN to ascii lmao
            if (ascii != 0):
                return ascii
        except (TypeError, IndexError):
            return default

    return default

def visualize_matrix(matrix : numpy.ndarray):
    # Custom colors
    csutom_cmap = ListedColormap(["white", "orange"])
    plt.imshow(matrix, cmap=csutom_cmap, interpolation='nearest')

    # Labels
    plt.xlabel("Row")
    plt.ylabel("Time")

    # Legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='0'),
        Patch(facecolor='orange', edgecolor='black', label='1')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the graph
    plt.show()

# --------------------
# Entry Point
if __name__ == "__main__":
    main()