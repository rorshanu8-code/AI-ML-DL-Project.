import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

# --------------------------------------------------
# Handwritten Character Recognition using CNN
# Dataset: EMNIST Balanced
# --------------------------------------------------

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load EMNIST Balanced dataset
print("Loading EMNIST Balanced dataset...")

(train_ds, test_ds), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Number of classes
num_classes = ds_info.features["label"].num_classes

print("Number of classes:", num_classes)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess(image, label):
    # Convert image to float
    image = tf.cast(image, tf.float32)

    # Normalize pixels from 0-255 to 0-1
    image = image / 255.0

    # Add channel dimension
    image = tf.expand_dims(image, axis=-1)

    return image, label


train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# Batch the dataset
# --------------------------------------------------

BATCH_SIZE = 128

train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)


# --------------------------------------------------
# Create CNN Model
# --------------------------------------------------

model = tf.keras.Sequential([

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Reduce image size
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce image size again
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps into one-dimensional data
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    tf.keras.layers.Dropout(0.3),

    # Output layer
    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# Display Model Information
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# Train the Model
# --------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds
)


# --------------------------------------------------
# Test Model Accuracy
# --------------------------------------------------

print("\nTesting the model...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)


# --------------------------------------------------
# SAVE THE TRAINED MODEL
# --------------------------------------------------

model_path = BASE_DIR / "handwritten_character_model.keras"

model.save(model_path)

print("\n----------------------------------------")
print("Model saved successfully!")
print("Model location:")
print(model_path)
print("----------------------------------------")