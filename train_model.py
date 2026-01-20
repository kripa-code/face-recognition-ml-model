import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Path to dataset
DATASET_DIR = "dataset"

faces = []    # face data
labels = []   # person names

# Read dataset
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Convert image to 1D array
        img_flatten = img.flatten()

        faces.append(img_flatten)
        labels.append(person_name)

# Convert lists to arrays
faces = np.array(faces)
labels = np.array(labels)

print("Total faces:", faces.shape[0])

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(faces, labels)

# Save trained model
with open("face_recognition_model_3.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Model trained and saved successfully!")