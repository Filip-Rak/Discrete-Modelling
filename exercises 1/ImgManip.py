from PIL import Image

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

def binarize(image, threshold = 127):
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
                pixel_value = 255 # White
            else:
                pixel_value = 0 # Black

            # Put adjusted pixel in the image
            gray_image.putpixel((x, y), pixel_value)

    return gray_image


# --------------------
# Helper Functions
def clamp(number, min = 0, max = 255):
    if number > max: return max
    if number < min: return min

    return number



