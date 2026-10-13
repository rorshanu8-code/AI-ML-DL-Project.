import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path


# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "handwritten_character_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# LABELS
# ==========================================

labels = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"
]


# ==========================================
# PREPARE ONE CHARACTER
# SAME AS WORKING SINGLE VERSION
# ==========================================

def prepare_character(image):

    img = np.array(image)

    coords = np.argwhere(img > 20)

    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[y_min:y_max + 1, x_min:x_max + 1]

    height, width = cropped.shape
    size = max(height, width)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = cropped

    char_img = Image.fromarray(square)

    char_img = char_img.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    final_image.paste(
        char_img,
        (4, 4)
    )

    # EMNIST orientation correction
    final_image = np.array(final_image).T

    image_array = (
        final_image.astype("float32") / 255.0
    )

    image_array = image_array.reshape(
        1, 28, 28, 1
    )

    return image_array


# ==========================================
# FIND CHARACTER BOXES
# ==========================================

def find_character_boxes():

    img = np.array(drawing_image)

    # Make binary image
    binary = np.where(
        img > 20,
        255,
        0
    ).astype(np.uint8)

    # Check whether anything is drawn
    if cv2.countNonZero(binary) == 0:
        return []

    # Slight dilation to connect strokes belonging
    # to the same handwritten character
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    processed = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        processed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        # Very low filtering so handwriting isn't rejected
        if area > 50 and h > 10:

            boxes.append(
                (x, y, w, h)
            )

    # Left-to-right order
    boxes.sort(
        key=lambda box: box[0]
    )

    return boxes


# ==========================================
# GET CHARACTER IMAGES
# ==========================================

def get_character_images():

    img = np.array(drawing_image)

    boxes = find_character_boxes()

    if not boxes:
        return []

    characters = []

    for x, y, w, h in boxes:

        padding = 5

        left = max(
            0,
            x - padding
        )

        top = max(
            0,
            y - padding
        )

        right = min(
            CANVAS_WIDTH,
            x + w + padding
        )

        bottom = min(
            CANVAS_HEIGHT,
            y + h + padding
        )

        character = img[
            top:bottom,
            left:right
        ]

        if np.count_nonzero(character) > 10:

            character_image = Image.fromarray(
                character.astype(np.uint8)
            )

            characters.append(
                character_image
            )

    return characters


# ==========================================
# PREDICT
# ==========================================

def predict_multiple():

    characters = get_character_images()

    if len(characters) == 0:

        result_label.config(
            text="Please draw characters first."
        )

        return

    result = ""
    confidence_values = []

    for character_image in characters:

        prepared = prepare_character(
            character_image
        )

        if prepared is None:
            continue

        prediction = model.predict(
            prepared,
            verbose=0
        )[0]

        index = int(
            np.argmax(prediction)
        )

        confidence = (
            float(prediction[index]) * 100
        )

        result += labels[index]

        confidence_values.append(
            confidence
        )

    if result == "":

        result_label.config(
            text="Please draw characters again."
        )

        return

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    result_label.config(
        text=(
            f"Prediction: {result}\n"
            f"Characters detected: {len(result)}\n"
            f"Average confidence: "
            f"{average_confidence:.2f}%"
        )
    )


# ==========================================
# CLEAR
# ==========================================

def clear_canvas():

    canvas.delete("all")

    drawing_image.paste(
        0,
        (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    )

    result_label.config(
        text="Draw multiple characters"
    )


# ==========================================
# DRAWING
# ==========================================

last_x = None
last_y = None


def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y

    radius = BRUSH_SIZE // 2

    canvas.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white"
    )

    drawing_draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )


def draw(event):

    global last_x, last_y

    if last_x is not None:

        canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        drawing_draw.line(
            [
                (last_x, last_y),
                (event.x, event.y)
            ],
            fill=255,
            width=BRUSH_SIZE
        )

        radius = BRUSH_SIZE // 2

        drawing_draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ),
            fill=255
        )

        last_x = event.x
        last_y = event.y


def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


# ==========================================
# EXIT
# ==========================================

def exit_program(event=None):

    root.destroy()


# ==========================================
# SETTINGS
# ==========================================

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 350
BRUSH_SIZE = 20


# ==========================================
# WINDOW
# ==========================================

root = tk.Tk()

root.title(
    "Multi-Character Handwritten Recognition"
)

root.attributes(
    "-fullscreen",
    True
)

root.configure(
    bg="#202020"
)


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="Multi-Character Handwritten Recognition",
    font=("Arial", 30, "bold"),
    bg="#202020",
    fg="white"
)

title.pack(pady=20)


# ==========================================
# INSTRUCTION
# ==========================================

instruction = tk.Label(
    root,
    text="Draw characters from left to right with a clear gap",
    font=("Arial", 18),
    bg="#202020",
    fg="white"
)

instruction.pack(pady=5)


# ==========================================
# CANVAS
# ==========================================

canvas = tk.Canvas(
    root,
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    bg="black",
    highlightthickness=3
)

canvas.pack(pady=20)


# ==========================================
# IMAGE
# ==========================================

drawing_image = Image.new(
    "L",
    (
        CANVAS_WIDTH,
        CANVAS_HEIGHT
    ),
    0
)

drawing_draw = ImageDraw.Draw(
    drawing_image
)


# ==========================================
# MOUSE
# ==========================================

canvas.bind(
    "<Button-1>",
    start_draw
)

canvas.bind(
    "<B1-Motion>",
    draw
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_draw
)


# ==========================================
# BUTTONS
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#202020"
)

button_frame.pack(pady=15)


predict_button = tk.Button(
    button_frame,
    text="PREDICT",
    command=predict_multiple,
    font=("Arial", 18, "bold"),
    width=12,
    height=2
)

predict_button.pack(
    side="left",
    padx=15
)


clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_canvas,
    font=("Arial", 18, "bold"),
    width=12,
    height=2
)

clear_button.pack(
    side="left",
    padx=15
)


exit_button = tk.Button(
    button_frame,
    text="EXIT",
    command=exit_program,
    font=("Arial", 18, "bold"),
    width=12,
    height=2
)

exit_button.pack(
    side="left",
    padx=15
)


# ==========================================
# RESULT
# ==========================================

result_label = tk.Label(
    root,
    text="Draw multiple characters",
    font=("Arial", 26, "bold"),
    bg="#202020",
    fg="white"
)

result_label.pack(pady=20)


# ==========================================
# ESCAPE
# ==========================================

root.bind(
    "<Escape>",
    exit_program
)


# ==========================================
# START
# ==========================================

root.mainloop()