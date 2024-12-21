import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("my_mnist_model.h5")

# Load and preprocess your custom handwritten digit
def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))                # Resize to 28x28 pixels
    img = np.array(img)                       # Convert to a NumPy array
    img = 255 - img                           # Invert colors (MNIST digits are white on black)
    img = img / 255.0                         # Normalize to [0, 1]
    img = img.reshape(1, 28, 28)              # Add batch dimension (1, 28, 28)
    return img

# Path to your custom image
image_path = "download.jpg"
processed_image = preprocess_image(image_path)

# Predict the digit
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction)  # Get the class with the highest probability

print(f"Predicted digit: {predicted_class}")
