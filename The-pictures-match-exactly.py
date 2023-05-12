import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('D:/sexual exploitation/my_model.h5')

image_size = (150, 150) # 更改為 (150, 150)
batch_size = 32

# Load image dataset C
c_data = keras.preprocessing.image_dataset_from_directory(
    'D:/sexual exploitation/C-victim',
    batch_size=batch_size,
    image_size=image_size,
    subset=None, # 修改 subset 參數
    seed=123 # 添加 seed 參數
)

# Load image dataset D
d_data = keras.preprocessing.image_dataset_from_directory(
    'D:/sexual exploitation/D-data',
    batch_size=batch_size,
    image_size=image_size,
    subset=None, # 修改 subset 參數
    seed=456 # 添加 seed 參數+
)

# Normalize the image data
# Normalize the image data
c_data = c_data.map(lambda x, y: (tf.image.resize(x, image_size) / 255.0, y))
d_data_normalized = d_data.map(lambda x, y: (tf.image.resize(x, image_size) / 255.0, y)) # Create a new dataset for normalized d_data

# Use the model to predict images in C and D
c_predictions = model.predict(c_data)
d_predictions = model.predict(d_data_normalized)

# Find matching images in D for each image in C
matching_indices = []
for i in range(len(c_predictions)):
    for j in range(len(d_predictions)):
        if np.array_equal(c_predictions[i], d_predictions[j]):
            matching_indices.append(j)

    # Print all matching image file names for current image in C
    if len(matching_indices) > 0:
        matching_file_names = [d_data.file_paths[index] for index in matching_indices]
        print(f"Found matching images in D with file names {matching_file_names} for image at index {i}.")
    else:
        print(f"No matching image found in D for image at index {i}.")

    # Reset matching indices list for next iteration
    matching_indices = []
