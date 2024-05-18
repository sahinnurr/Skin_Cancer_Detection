from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
import pandas as pd
import shutil
import splitfolders
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Defined data generators with preprocessing and augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Apply normalization
    rotation_range=20,  # Rotate images randomly by up to 20 degrees
    width_shift_range=0.1,  # Shift width by up to 10%
    height_shift_range=0.1,  # Shift height by up to 10%
    horizontal_flip=True,  # Flip horizontally
    zoom_range=0.1  # Zoom by up to 10%

    )


train_dir = os.getcwd() + "/cell_data_split/train/"
val_dir = os.getcwd() + "/cell_data_split/val/"
test_dir = os.getcwd() + "/cell_data_split/test/"

# Train data generator
train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical', #Indicates that the labels are provided as categorical (one-hot encoded) vectors.
                                         batch_size=16,  #set of input data to process together at the same time
                                         target_size=(224,224))  #Resize images)

# Validation data generator
val_data_keras = datagen.flow_from_directory(
    directory=val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Testing data generator
test_data_keras = datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Explore dataset structure

data_dir = os.path.join(os.getcwd(), "dataset", "reorganized")
skin_df = pd.read_csv('dataset/HAM10000_metadata.csv')

# Calculate the number of unique classes
num_classes = len(skin_df['dx'].unique())
print("Number of classes:", num_classes)


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

#Spliting the dataset into training (70%), validation (20%), and test (10%) sets.
input_folder = os.getcwd() + "/dataset/reorganized/"

splitfolders.ratio(input_folder, output="cell_data_split",
                   seed=42, ratio=(.7, .2, .1),
                   group_prefix=None)   # default values



#Creating the model
model_path = 'skin_cancer_detection_model.keras'

if os.path.exists(model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    print("Loaded pre-trained model.")

    # Print model summary
    model.summary()




else:
    # Load EfficientNet model with pre-trained weights
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)  # Add a dense layer for more representation
    output = Dense(num_classes, activation='softmax')(x)  # Adjust num_classes to your dataset

    # Create model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Training the model and evaluating performance metrics on the validation set if the model is not exist
    epochs = 20
    history = model.fit(
        train_data_keras,
        epochs=epochs,
        validation_data=val_data_keras,
    )

    # Saving the model
    model.save('skin_cancer_detection_model.keras')
    print("Model saved.")
# Evaluate the trained model on the test dataset
test_loss, test_accuracy = model.evaluate(test_data_keras)

# Generate predictions for the test dataset
predictions = model.predict(test_data_keras)
predicted_labels = predictions.argmax(axis=1)

# Get the true labels for the test dataset
true_labels = test_data_keras.classes

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Print the evaluation metrics
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)













