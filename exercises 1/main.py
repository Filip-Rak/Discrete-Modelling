from ImgManip import set_brightness, binarize
from PIL import Image, ImageTk
from tkinter import ttk
import tkinter

# Main Functions
# --------------------
def main():
    # Setup
    img = Image.open("Image/Input/inp_img.jpg")

    # Excersises
    ex1(img)
    ex2(img)
    ex3(img)
    ex4(img)

# Exercise Functions
def ex1(img):
    # Adjust the brightness of the image and save it
    new_img = set_brightness(img, 0.2)
    new_img.save("Image/Output/ex1_out.png")

def ex2(img):
    # Adjust brightness incrementally and save each result
    new_img = img.copy()
    for i in range(1, 4):
        new_img = set_brightness(new_img, 1.2)
        new_img.save("Image/Output/ex2_%d_out.png" % (i))

def ex3(img):
    # Save the result of binarization (default threshold = 127)
    binarize(img).save("Image/Output/ex3_out.png")

def ex4(img):
    # Add UI for changing the threshold of binarization and saving the image
    # Colors
    color1 = "#1E3E62"  #BG1
    color2 = "WHITE"  #FG
    color3 = "#0B192C"  #BG2

    # Set window
    window = tkinter.Tk()
    window.title("Binarization")
    window.configure(bg = color3)
    # window.state('zoomed')

    # Set label for image display
    img_label = ttk.Label(window)
    img_label.grid(padx=10, pady=10)
    img_label.binary_image = img

    # Create slider for the threshold
    slider_cmd = lambda value: update_image(img, value, img_label)
    scale = tkinter.Scale(window, from_=0, to=255, orient='horizontal', label='THRESHOLD', bg=color1, fg=color2, command=slider_cmd)
    scale.grid(row=0, column=1)
    scale.set(127)  # Set default threshold

    # Create save button
    button_cmd = lambda: img_label.binary_image.save("Image/Output/ex4_out.png")
    button = tkinter.Button(window, text='SAVE', bg=color1, fg=color2, command=button_cmd)
    button.grid(row=0, column=2, padx=10)

    # Set the initial image
    update_image(img, scale.get(), img_label)

    # Start the GUI main loop
    window.mainloop()

# Helper Functions
# --------------------
def update_image(img, threshold, img_label):
    # Save the binarized image
    threshold = int(threshold)
    binary_image = binarize(img, threshold)
    img_label.binary_image = binary_image

    # Convert binary_image to PhotoImage
    img_display = ImageTk.PhotoImage(binary_image)

    # Update the label with the new PhotoImage
    img_label.image = img_display # Keep a reference to prevent garbage collection
    img_label.configure(image = img_display)  # Update the image in the label

# Entry Point
# --------------------
if __name__ == "__main__":
    main()