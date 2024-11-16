from Automaton import CellularAutomaton, CellState
from Visualization import Visualization
from PIL import Image
import numpy

# Constants
# --------------------
InputPath = "Input/"

# Helper Functions
# --------------------
def load_map_from_image(image_path: str, rows: int, cols: int) -> numpy.ndarray:
    """
    Load a map from an image file. \n
    Resizes the image to fit the size of the grid and maps states based on color.
    """
    
    # Create grid
    img = Image.open(image_path).convert("RGB")
    img = img.resize((cols, rows))  # Image scailing
    grid = numpy.zeros((rows, cols), dtype=int)

    # Map pixels
    for y in range(rows):
        for x in range(cols):
            pixel = img.getpixel((x, y))
            grid[y, x] = map_pixel_to_state(pixel)

    return grid

def map_pixel_to_state(pixel: float) -> int:
    """Returns a state assgined to a given pixel based on it's color."""
    r, g, b = pixel

    # Water: Blue dominance
    if b > g and b > r and b - max(r, g) > 20:
        return CellState.WATER.value

    # Forest: Green dominance
    if g > r and g > b:
        if g > 100:  # Brighter green
            return CellState.FOREST.value
        else:  # Darker green
            return CellState.OVERGORWN_FOREST.value

    # Rest considered as ground
    return CellState.EMPTY.value

# Main function
# --------------------
def main():
    # Settings
    rows = 70
    cols = 120
    cell_size = 10
    use_map = True  # Toggle for random or image map based grid

    # Create cellular automaton
    automaton = CellularAutomaton(rows, cols)

    if use_map:
        map_path = InputPath + "photo1.png"
        grid = load_map_from_image(map_path, rows, cols)
        automaton.initialize_from_map(grid)
    else:
        # Initialize with random states
        state_pool = [CellState.FOREST.value, CellState.OVERGORWN_FOREST.value]
        automaton.initialize_from_map(numpy.random.choice(state_pool, size=(rows, cols)))

    # Create visualization
    visualization = Visualization(automaton, cell_size)

    # Run visualization
    visualization.run()

# Entry Point
# --------------------
if __name__ == "__main__":
    main()
