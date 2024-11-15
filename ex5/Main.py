from Automaton import CellularAutomaton
from Visualization import Visualization
import numpy as np

def load_map_from_image(image_path, cell_size):
    """
    Load a map from an image file and convert it into a grid.
    White pixels (255,255,255): Empty
    Green pixels (0,255,0): Forest
    Red pixels (255,0,0): Fire
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((img.width // cell_size, img.height // cell_size))
    grid = np.zeros((img.height, img.width), dtype=int)

    for y in range(img.height):
        for x in range(img.width):
            pixel = img.getpixel((x, y))
            if pixel == (255, 255, 255):  # Empty
                grid[y, x] = 0
            elif pixel == (0, 255, 0):  # Forest
                grid[y, x] = 3
            elif pixel == (255, 0, 0):  # Fire
                grid[y, x] = 1
            else:  # Default to empty
                grid[y, x] = 0

    return grid

def main():
    # Settings
    rows = 70
    cols = 120
    cell_size = 10
    use_map = False  # Set to True to load from an image

    # Create cellular automaton
    automaton = CellularAutomaton(rows, cols)

    if use_map:
        map_path = "map.png"  # Replace with the path to your map image
        grid = load_map_from_image(map_path, cell_size)
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
