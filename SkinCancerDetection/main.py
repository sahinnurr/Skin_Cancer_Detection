from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
import pandas as pd
import shutil
import splitfolders
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define the preprocessing function
def preprocess_image(img):
    img = img.astype('float32')  # Convert to float32
    img /= 255.0  # Normalize pixel values to [0, 1]
    return img

# Define your data generators with preprocessing and augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Apply normalization
    rotation_range=20,  # Rotate images randomly by up to 20 degrees
    width_shift_range=0.1,  # Shift width by up to 10%
    height_shift_range=0.1,  # Shift height by up to 10%
    horizontal_flip=True,  # Flip horizontally
    zoom_range=0.1  # Zoom by up to 10%
)


train_dir = os.getcwd() + "/dataset/reorganized/"

train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical', #Indicates that the labels are provided as categorical (one-hot encoded) vectors.
                                         batch_size=16,  #set of input data to process together at the same time
                                         target_size=(224,224))  #Resize images

# Explore dataset structure

data_dir = os.path.join(os.getcwd(), "dataset", "reorganized")
skin_df = pd.read_csv('dataset/HAM10000_metadata.csv')

print(skin_df.head()) # Examine the beginning of the dataset
print(skin_df.info())  # Getting general information of the dataset
print(skin_df['dx'].value_counts())

# Show sample images
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    img_id = skin_df.iloc[i]['image_id']
    img_path = os.path.join(data_dir, skin_df.iloc[i]['dx'], img_id + ".jpg")
    if os.path.exists(img_path):  # Check if the file exists before loading
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
    else:
        print("File not found:", img_path)  # Print a message if the file is not found

plt.show()

input_folder = os.getcwd() + "/dataset/reorganized/"

splitfolders.ratio(input_folder, output="cell_data_split",
                   seed=42, ratio=(.7, .2, .1),
                   group_prefix=None)   # default values









