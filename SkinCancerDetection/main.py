import os
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input


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
                                         class_mode='categorical',
                                         batch_size=16,
                                         target_size=(224, 224))

#create skin Cancer Detection model
efficientnet_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
efficientnet_model = hub.KerasLayer(efficientnet_url, trainable=False) #applies feature extractors of the pre-trained EfficientNet model on a given image input.

#defines the architecture of the model
num_classes = len(train_data_keras.class_indices)  # Number of unique classes
model = models.Sequential([
    efficientnet_model,
    layers.Dense(num_classes, activation='softmax')  #takes the input feature vector and produces outputs determine which class this feature vector belongs to
])



# Compile the model with an appropriate loss function and optimizer.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Explore dataset structure
data_dir = os.path.join(os.getcwd(), "dataset", "reorganized")
skin_df = pd.read_csv('dataset/HAM10000_metadata.csv')

print(skin_df.head())  # Examine the beginning of the dataset
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

# Split the data into train, validation, and test sets
input_folder = os.getcwd() + "/dataset/reorganized/"
splitfolders.ratio(input_folder, output="cell_data_split",
                   seed=42, ratio=(.7, .2, .1),
                   group_prefix=None)  # default values
