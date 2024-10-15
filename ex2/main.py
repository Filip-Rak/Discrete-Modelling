from PIL import Image
from ImgManip import *
import numpy

# --------------------
# Constants
INPUT_PATH = "Image/Input/"
OUTPUT_PATH = "Image/Output/"


# --------------------
# Main Function
def main():
    # Load input images
    inp_img_1 = Image.open(INPUT_PATH + "inp_img.jpg")

    # Excersises
    # ex_1(inp_img_1)
    ex_2(inp_img_1)

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
    image.save(OUTPUT_PATH + "out_1_binarized.jpg")
    out_1.save(OUTPUT_PATH + "out_1_1.jpg")
    out_2.save(OUTPUT_PATH + "out_1_2.jpg")
    out_3.save(OUTPUT_PATH + "out_1_3.jpg")

def ex_2(img):
    mask = numpy.array(
        [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]
    )

    im_masked = apply_convolusion(img, mask)
    im_masked.save(OUTPUT_PATH + "out_2_1.jpg")

# --------------------
# Entry Point
if __name__ == "__main__":
    main()