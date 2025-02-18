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
│── imgManip.py         # Python source code with image manipulation functions
│── main.py             # Entry point for the program and tasks within this exercise
```

## Exercise 3
Elementary cellular automaton in Python with usage of NumPy and Matplotlib libraries:
- **Border Handling**: Management of edge cases in automaton evolution.
- **Seed Generation**: Initialization of starting patterns.
- **Rule Application**: Implementing of rule sets for evolution.
- **Binary Rule Encoding** – Each rule  is represented as an 8-bit binary number, where each bit corresponds to the output state.
- **Progress Visualization** - Matplotlib graph visulizing the progress of automaton.
- **Output to File** - Progress of the automaton is saved to .csv file.

![Graph visualizing the progress of the automaton with random seed and rule](Media/ex3_graph.png)

*Graph visualizing the progress of the automaton with random seed and rule*

### Directory Structure
```
ex3/  
│── Output/             # Stores automaton progress in .csv format
│── elementaryCa.py     # Python source code with automaton functions
│── main.py             # Entry point for the program and tasks within this exercise
```

## Exercise 4
Implementation of Conway's Game of Life in Python including:
- **Definition of various patterns**.
- **Rule Application** with accordance to Game of Life.
- **Hadnling of boundary conditions**.
- **Grid-based simulation** of cell evolution.
- **Export of results as gifs** for visualization.

![Gif visualizing the progress of 2D automaton with random seed](Media/ex4_automaton.gif)

*Gif visualizing the progress of 2D automaton with random seed*

### Directory Structure
```
ex4/  
│── Output/				# Results of various tasks saved as gifs
│   │── good rands/		# Handpicked interesting random results
│
│── GameOfLife.py		# Python source code with automaton functions
│── Main.py				# Entry point for the program and tasks within this exercise
```


