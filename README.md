# Overview
This repository contains solutions to various exercises for the **Discrete Modelling** course during the fifth semester at **AGH University of Krakow**. The exercises involve implementing different computational models and simulations in **Python** and **C++**, covering topics such as **image processing**, **cellular automata**, and **fluid simulations**.

![Forest Fire Simulation](Media/ex5_forest_fire.gif)

*Forest fire simulation*

---

# Main Repository Structure
```
Discrete-Modelling/
│── Coursework-Reports/  # Contains reports for exercises
│── Media/               # Media for README.md file
│── ex1/                 # Basic image manipulation (Python)
│── ex2/                 # Image processing with convolution (Python)
│── ex3/                 # Elementary Cellular Automaton (Python)
│── ex4/                 # Game of Life (Python, GIF export)
│── ex5/                 # Forest Fire Simulation (Python, PyGame, Interactive map)
│── ex6/                 # LGA Automaton (C++, SFML/TGUI visualization)
│── ex7/                 # LBM Gas Simulation (C++, extended framework)
│── README.md            # This file
```
---

# Exercises
## Exercise 1
Basic image manipulation in Python with the usage of Pillow library:
- **Brightness adjustment**: Modifying image brightness levels.
- **Binarization**: Converting images to black and white based on a threshold.
- **Border Handling**: Managing edges when applying transformations.

![Comparison between initial image and binarized version](Media/ex1_binarization.png)

*Comparison between initial image and binarized version*

### Directory Structure
```
ex1/  
│── Image/
│   │── Input/          # Initial images
│   └── Output/         # Results
│
│── imgManip.py         # Python source code with image manipulation functions
│── main.py             # Entry point for the program and tasks within this exercise
```

## Exercise 2
Further image processing making use of previous functionality and NumPy library:
- **Morphological Transformations**: Applying opening and closing operations to clean up images.
- **Convolution**: Applying kernels for effects like blurring and edge detection.

![Comparison between initial, dilated and eroded image](Media/ex2_comparison.png)

*Comparison between initial, dilated and eroded image*

### Directory Structure
```
ex2/  
│── Images/
│   │── Input/          # Initial images
│   └── Output/         # Results
│
│── Masks/              # Text files defining kernels for image processing
│── imgManip.py         # Python source code with image manipulation
│── main.py             # Entry point for the program and tasks within this exercise
```

