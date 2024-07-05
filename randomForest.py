import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load MNIST dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images for RandomForest
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

# Define and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(x_train_flat, y_train)

# # Evaluate the model
y_pred = rf_model.predict(x_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"RandomForest Accuracy: {accuracy}")

# # Save the model
rf_model_save_path = 'random_forest_model.pkl'
joblib.dump(rf_model, rf_model_save_path)
print("DONE!!!!")
# Load the model
rf_model = joblib.load(rf_model_save_path)

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
            digit_flat = digit.reshape(1, -1)
            predicted_digit = rf_model.predict(digit_flat)[0]
            print(f"Digit {i + 1} is probably a {predicted_digit}")
            
            axes[i].imshow(digit, cmap=plt.cm.binary)
            axes[i].set_title(f"Predicted: {predicted_digit}")
            axes[i].axis('off')
        
        plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
        image_number += 1
