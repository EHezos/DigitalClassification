import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Save the model
model_save_path = 'handwritten_model.h5'
model.save(model_save_path)

# Load the model
model = tf.keras.models.load_model(model_save_path)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    # Extract digit images
    digit_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        digit = img[y:y+h, x:x+w]
        
        # Calculate padding
        h_padding = max((28 - h) // 2, 0)
        w_padding = max((28 - w) // 2, 0)
        
        # Add padding to maintain aspect ratio
        digit = cv2.copyMakeBorder(digit, h_padding, 28-h-h_padding, w_padding, 28-w-w_padding, cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 28x28
        digit = cv2.resize(digit, (28, 28))
        digit = digit / 255.0
        digit_images.append(digit)
    
    return digit_images

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        digit_images = preprocess_image(f"digits/digit{image_number}.png")
        
        fig, axes = plt.subplots(1, len(digit_images))
        if len(digit_images) == 1:
            axes = [axes]
        
        for i, digit in enumerate(digit_images):
            digit = np.expand_dims(digit, axis=0)
            prediction = model.predict(digit)
            predicted_digit = np.argmax(prediction)
            print(f"Digit {i + 1} is probably a {predicted_digit}")
            
            axes[i].imshow(digit[0], cmap=plt.cm.binary)
            axes[i].set_title(f"Predicted: {predicted_digit}")
            axes[i].axis('off')
        
        plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
        image_number += 1
