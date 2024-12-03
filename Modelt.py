# Import Necessary Libraries
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

# Load and Preprocess Dataset
def load_images(image_dir, target_size=(224, 224)):
    """
    Load and preprocess images from a directory.
    :param image_dir: Path to the directory containing images.
    :param target_size: Target size for resizing images.
    :return: Preprocessed images as a NumPy array.
    """
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = plt.imread(os.path.join(image_dir, filename))
            img_resized = resize(img, target_size, anti_aliasing=True)
            images.append(img_resized)
    images = np.array(images)
    images = images / 255.0  # Normalize pixel values
    return images

# Path to dataset
image_path = r'C:\Users\mutha\Desktop\Heart\data'

# Load images
images = load_images(image_path)

# Split data into training and validation sets
x_train, x_val = train_test_split(images, test_size=0.2, random_state=42)

# Define Autoencoder Model
def create_autoencoder(input_shape):
    """
    Create an autoencoder model for one-class classification.
    :param input_shape: Shape of input images.
    :return: Compiled autoencoder model.
    """
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Define input shape
input_shape = (224, 224, 3)  # Adjust dimensions based on your images

# Create the autoencoder
autoencoder = create_autoencoder(input_shape)

# Train the Autoencoder
history = autoencoder.fit(
    x_train, x_train,
    validation_data=(x_val, x_val),
    epochs=20,  # Adjust as needed
    batch_size=32,  # Adjust as needed
    verbose=1
)

# Save the Model
model_save_path = r"C:\Users\mutha\Desktop\Heart\heart_disease_autoencoder.h5"
autoencoder.save(model_save_path)
print(f"Model saved as {model_save_path}")

# Evaluate the Model
loss = autoencoder.evaluate(x_val, x_val)
print(f"Validation Loss: {loss:.4f}")

# Visualize Reconstruction Performance
def visualize_reconstruction(model, images, n=5):
    """
    Visualize the reconstruction of the autoencoder.
    :param model: Trained autoencoder model.
    :param images: Validation images.
    :param n: Number of samples to visualize.
    """
    reconstructed = model.predict(images[:n])
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original image
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.title("Original")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()

visualize_reconstruction(autoencoder, x_val)

# Anomaly Detection: Compute Reconstruction Error
def compute_reconstruction_error(model, images):
    """
    Compute reconstruction error for each image.
    :param model: Trained autoencoder model.
    :param images: Images to compute reconstruction error on.
    :return: Reconstruction errors.
    """
    reconstructed = model.predict(images)
    errors = np.mean(np.square(images - reconstructed), axis=(1, 2, 3))
    return errors

# Compute reconstruction errors
errors = compute_reconstruction_error(autoencoder, x_val)
threshold = np.percentile(errors, 95)  # Set threshold for anomaly detection
print(f"Threshold for anomaly detection: {threshold:.4f}")

# Detect anomalies (e.g., non-heart disease images)
anomalies = errors > threshold
print(f"Number of anomalies detected: {np.sum(anomalies)}")
