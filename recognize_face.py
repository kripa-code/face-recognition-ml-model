import cv2
import pickle
import numpy as np

# Load trained model
with open("face_recognition_model_3.pkl", "rb") as f:
    knn = pickle.load(f)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and resize face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        # Convert face to 1D array
        face_flatten = face.flatten().reshape(1, -1)

        # Predict name
        name = knn.predict(face_flatten)[0]

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()