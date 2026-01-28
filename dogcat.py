import tensorflow as tf
import pandas as pd
import os

df = pd.read_csv("mini-project/cat_dog/cat_dog.csv")

IMAGE_DIR = "mini-project/cat_dog/cat_dog"

df["image_path"] = df["image"].apply(
    lambda x: os.path.join(IMAGE_DIR, x)
)

IMG_SIZE = 128

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

paths = df["image_path"].values
labels = df["labels"].values

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

train_size = int(0.8 * len(df))

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

import numpy as np

img_path = "mini-project/cat_dog/cat_dog/cat.347.jpg"

img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (128,128))
img = img / 255.0
img = tf.expand_dims(img, axis=0)

pred = model.predict(img)

print("Dog 🐶" if pred[0][0] > 0.5 else "Cat 🐱")
