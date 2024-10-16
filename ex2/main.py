from PIL import Image
from ImgManip import *
import numpy

# --------------------
# Constants
INPUT_PATH = "Images/Input/"
OUTPUT_PATH = "Images/Output/"
MASK_PATH = "Masks/"

# --------------------
# Main Function
def main():
    # Load input images
    inp_img_1 = Image.open(INPUT_PATH + "inp_img1.jpg")
    inp_img_1_2 = Image.open(INPUT_PATH + "inp_img1_2.jpg")
    inp_img_2 = Image.open(INPUT_PATH + "inp_img2.jpg")

    # Excersises
    ex_1(inp_img_1)
    ex_2(inp_img_1_2, inp_img_2)

# --------------------
# Excersises
def ex_1(img):
    # Prepare the image
    image = binarize(img)

    # Get transformations
    out_1 = morphological_transform(image, Operation.dilate, 1, BorderHandling.ZERO_PADDING)
    out_2 = morphological_transform(image, Operation.erode, 3, BorderHandling.WRAP_AROUND)
    out_3 = morphological_closing(image, 3, BorderHandling.EDGE_PADDING)

    # Save results
    image.save(OUTPUT_PATH + "out_1_binarized.png")
    out_1.save(OUTPUT_PATH + "out_1_1.png")
    out_2.save(OUTPUT_PATH + "out_1_2.png")
    out_3.save(OUTPUT_PATH + "out_1_3.png")

def ex_2(img1, img2):
    gauss3x3 = load_mask(MASK_PATH + "Gauss3x3.txt")
    wf5x5 = load_mask(MASK_PATH + "WarpedFocus5x5.txt")
    apply_convolusion(img1, gauss3x3).save(OUTPUT_PATH + "out_2_1-Gauss3x3.png")
    apply_convolusion(img2, wf5x5).save(OUTPUT_PATH + "out_2_2-Alc5x5.png")

# Helper Functions
# --------------------
def load_mask(filename, delimiter = ","):
    with open(filename, 'r') as file:
        matrix = [[int(num) for num in line.split(delimiter)] for line in file]

    # Convert to a numpy array
    return numpy.array(matrix, dtype=numpy.float32)

# --------------------
# Entry Point
if __name__ == "__main__":
    main()