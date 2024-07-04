import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x = mnist.data
y = mnist.target.astype(np.int32)

# Normalize the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Create and train the Random Forest model
rf = RandomForestClassifier(n_estimators=40, random_state=42)
rf.fit(x_train, y_train)

# Evaluate the model
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict on custom images
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        img = cv2.resize(img, (28, 28))  # Resize image to 28x28 pixels
        img = np.invert(img)  # Invert colors (black background, white digits)
        img = img.reshape(1, -1)  # Flatten the image to a vector
        
        img = scaler.transform(img)  # Normalize the image using the same scaler

        prediction = rf.predict(img)
        print(f"This digit is probably a {prediction[0]}")
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
    finally:
        image_number += 1
