from enum import Enum
from PIL import Image
import numpy

# --------------------
# Constants
PIXEL_MAX = 255 # White
PIXEL_MIN = 0   # Black
SPECIAL_VALUE = -1   # Black

# --------------------
# Helper Classes
class BorderHandling(Enum):
    ZERO_PADDING = 1
    EDGE_PADDING = 2
    REFLECT_PADDING = 3
    WRAP_AROUND = 4

class Operation(Enum):
    dilate = 1
    erode = 2

# --------------------
# Helper Functions
def handle_border(image_matrix, x, y, nx, ny, border_handling):
    # Get position inside the image (i)
    ix = x + nx
    iy = y + ny

    # Save image shape
    rows = image_matrix.shape[0]
    cols = image_matrix.shape[1]

    # Zero Padding: Lacking pixels are treated as 0s
    if border_handling == BorderHandling.ZERO_PADDING:
        # If outside, return as 0
        if ix < 0 or ix >= rows or iy < 0 or iy >= cols:
            return SPECIAL_VALUE, PIXEL_MIN
        else:
            return ix, iy

    # Edge Padding: Extending the image border
    elif border_handling == BorderHandling.EDGE_PADDING:
        nx = max(0, min(ix, rows - 1))
        ny = max(0, min(iy, cols - 1))
        return nx, ny

    # Reflect Padding: Reflecting the image starting from the border
    elif border_handling == BorderHandling.REFLECT_PADDING:
        nx = reflect_padding(ix, rows)
        ny = reflect_padding(iy, cols)
        return nx, ny

    # Wrap-Around: Replicates the other side of the image
    elif border_handling == BorderHandling.WRAP_AROUND:
        nx = (ix) % rows
        ny = (iy) % cols
        return nx, ny

def clamp(number, min = PIXEL_MIN, max = PIXEL_MAX):
    if number > max: return max
    if number < min: return min

    return number

def reflect_padding(i, size):
    if i < 0:
        return -i
    elif i >= size:
        return size - 1 - (i - size) % size
    else:
        return i

# --------------------
# Exportable Functions
def set_brightness(image, factor):
    # Convert to RGB
    new_image = image.convert('RGB')
    width, height = new_image.size

    # Go through all the pixels
    for x in range(width):
        for y in range(height):
            # Get pixel
            r, g, b = new_image.getpixel((x, y))

            # Adjust brightness of the pixel
            r = clamp(int(r * factor))
            g = clamp(int(g * factor))
            b = clamp(int(b * factor))

            # Put adjusted pixel in image
            new_image.putpixel((x, y), (r, g, b))

    return new_image

def binarize(image, threshold=127):
    # Convert to grayscale
    gray_image = image.convert("L")
    width, height = gray_image.size

    # Go through all the pixels
    for x in range(width):
        for y in range(height):
            # Get the pixel
            pixel_value = gray_image.getpixel((x, y))

            # Set color based on threshold
            if pixel_value > threshold:
                pixel_value = PIXEL_MAX
            else:
                pixel_value = PIXEL_MIN

            # Put adjusted pixel in the image
            gray_image.putpixel((x, y), pixel_value)

    return gray_image

def morphological_transform(image, operation=Operation.dilate, radius=1, border_handling = BorderHandling.ZERO_PADDING):
    # Copy the original image as a number matrix
    img_matrix = numpy.array(image) / PIXEL_MAX

    # Create an array for an output
    rows, cols = img_matrix.shape
    output_matrix = numpy.zeros((rows, cols))

    # Set up the operation
    if operation == Operation.dilate:
        comparison_func = min
        initial_val = 1
    else:
        comparison_func = max
        initial_val = 0

    # Go through every pixel within the image
    for x in range(rows):
        for y in range(cols):
            result_value = initial_val;
            # Go through neighbours within radius
            for nx in range(-radius, radius + 1):
                for ny in range(- radius, radius + 1):
                    # Get neighbour's value, respecting set boundary condition
                    tx, ty = handle_border(img_matrix, x, y, nx, ny, border_handling)
                    if tx == SPECIAL_VALUE:
                        neighbour_value = ty
                    else:
                        neighbour_value = img_matrix[tx, ty]

                    result_value = comparison_func(result_value, neighbour_value)

            # Assign the result to the output matrix
            output_matrix[x, y] = result_value

    # Invert the pixel values to correct black and white inversion
    output_matrix = output_matrix * PIXEL_MAX

    # Convert output matrix to an image
    output_matrix = numpy.array(output_matrix, dtype=numpy.uint8)
    output_image = Image.fromarray(output_matrix)

    return output_image

def morphological_openning(image, radius=1, border_handling = BorderHandling.ZERO_PADDING):
    eroded_image = morphological_transform(image, Operation.erode, radius, border_handling)
    opened_image = morphological_transform(eroded_image, Operation.dilate, radius, border_handling)

    return opened_image

def morphological_closing(image, radius=1, border_handling = BorderHandling.ZERO_PADDING):
    dilated_image = morphological_transform(image, Operation.dilate, radius, border_handling)
    closed_image = morphological_transform(dilated_image, Operation.erode, radius, border_handling)

    return closed_image

def apply_convolusion(image, mask, border_handling = BorderHandling.ZERO_PADDING):
    # Convert the image to a matrix
    img_matrix = numpy.array(image, dtype=numpy.float32)

    # Get dimensions
    rows, cols, channels = img_matrix.shape
    mask_size = mask.shape[0]
    pad_size = mask_size // 2

    # Use mask's sum for normalization
    mask_sum = numpy.sum(mask)

    # Declare the output matrix and fill it with zeros
    output_matrix = numpy.zeros((rows, cols, channels), dtype=numpy.float32)

    # Go through every pixel
    for x in range(rows):
        for y in range(cols):
            # Save the total sum
            pixel_values = [0, 0 ,0]
            for lx in range(-pad_size, pad_size + 1):
                for ly in range(-pad_size, pad_size + 1):
                    # Get the pixel value with respect to boundary condition
                    tx, ty = handle_border(img_matrix, x, y, lx, ly, border_handling)

                    # Multiply each channel with cresponding channel from handle border
                    for i in range(0, channels):
                        pixel_values[i] += img_matrix[tx, ty, i] * mask[lx + pad_size, ly + pad_size]

            # Put the pixel in the matrix
            for i in range(0, channels):
                output_matrix[x, y, i] = pixel_values[i] / mask_sum

    # Clip results within safe values
    output_matrix = numpy.clip(output_matrix, PIXEL_MIN, PIXEL_MAX)

    # Convert output matrix to an image
    output_matrix = numpy.array(output_matrix, dtype=numpy.uint8)
    output_image = Image.fromarray(output_matrix)

    return  output_image

