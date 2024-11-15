from Automaton import CellularAutomaton, CellState
from Visualization import Visualization
import numpy as np

def load_map_from_image(image_path, rows, cols):
    """
    Load a map from an image file and map colors dynamically to states.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((cols, rows))  # Skalowanie obrazu
    grid = np.zeros((rows, cols), dtype=int)

    for y in range(rows):
        for x in range(cols):
            pixel = img.getpixel((x, y))
            grid[y, x] = map_pixel_to_state_dynamic(pixel)

    return grid

def map_pixel_to_state_dynamic(pixel):
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



INPUT_PATH = "Input/"

def main():
    # Settings
    rows = 70
    cols = 120
    cell_size = 10
    use_map = True  # Set to True to load from an image

    # Create cellular automaton
    automaton = CellularAutomaton(rows, cols)

    if use_map:
        map_path = INPUT_PATH + "map5.png"  # Replace with the path to your map image
        grid = load_map_from_image(map_path, rows, cols)
        automaton.initialize_from_map(grid)
    else:
        # Initialize with some random data
        automaton.initialize_from_map(np.random.choice([4, 5], size=(rows, cols)))

    # Create visualization
    visualization = Visualization(automaton, cell_size)

    # Run visualization
    visualization.run()

if __name__ == "__main__":
    main()
