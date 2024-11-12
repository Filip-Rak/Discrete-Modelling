from GameOfLife import *

# Constants
# --------------------
OUTPUT_PATH = "Output/"

# Main Functions
# --------------------
def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()

# Exercises
# --------------------
def ex1():  # Glider
    print("\n---------- Glider: Wrap Around ----------")
    initial_matrix = get_initial_state(10, 10, Pattern.GLIDER)
    result = apply_rules(initial_matrix, 100, BoundaryCondition.PERIODIC)
    save_as_gif(result, OUTPUT_PATH + "out1_1_glider_wrap.gif", 80, duration=20)

    print("\n---------- Glider: Mirror ----------")
    result = apply_rules(initial_matrix, 40, BoundaryCondition.REFLECTIVE)
    save_as_gif(result, OUTPUT_PATH + "out1_2_glider_mirror.gif", 80, duration=40)

def ex2():
    print("\n---------- Oscillator ----------")
    initial_matrix = get_initial_state(11, 11, Pattern.OSCILLATOR)
    result = apply_rules(initial_matrix, 100)
    save_as_gif(result, OUTPUT_PATH + "out2_1_oscillator.gif", 80, duration=100)

def ex3():
    print("\n---------- Still ----------")
    initial_matrix = get_initial_state(10, 10, Pattern.STILL)
    result = apply_rules(initial_matrix, 100)
    save_as_gif(result, OUTPUT_PATH + "out3_1_still.gif", 80, duration=20)

def ex4():
    print("\n---------- Random Small ----------")
    initial_matrix = get_initial_state(10, 10, Pattern.RANDOM)
    result = apply_rules(initial_matrix, 100)
    save_as_gif(result, OUTPUT_PATH + "out4_1_random_small.gif", 80, duration=40)

    print("\n---------- Random Big ----------")
    initial_matrix = get_initial_state(100, 100, Pattern.RANDOM, rand_x=50, rand_y=50)
    result = apply_rules(initial_matrix, 500)
    save_as_gif(result, OUTPUT_PATH + "out4_2_random_big.gif", 15, duration=20)

def ex5():
    print("\n---------- Random Huge ----------")
    initial_matrix = get_initial_state(500, 500, Pattern.RANDOM, rand_x=400, rand_y=400)
    result = apply_rules(initial_matrix, 2000)
    save_as_gif(result, OUTPUT_PATH + "out5_1_random_huge.gif", 3, duration=20)

# Entry Point
# --------------------
if __name__ == "__main__":
    main()