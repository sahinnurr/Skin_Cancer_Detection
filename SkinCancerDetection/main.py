from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
import pandas
import shutil


datagen = ImageDataGenerator()  #This is for data augmentation

train_dir = os.getcwd() + "/dataset/reorganized/"

train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical', #Indicates that the labels are provided as categorical (one-hot encoded) vectors.
                                         batch_size=16,  #set of input data to process together at the same time
                                         target_size=(32,32))  #Resize images












