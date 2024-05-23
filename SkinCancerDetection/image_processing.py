import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image to match the model's input requirements.

    :param image: The input image (PIL Image or NumPy array).
    :param target_size: The target size (width, height) for the image.
    :return: The preprocessed image (NumPy array).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Resize the image
    image = cv2.resize(image, target_size)

    # Convert the image to float32
    image = image.astype(np.float32)

    # Apply EfficientNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
