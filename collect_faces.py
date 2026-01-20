import cv2
import os

# Ask person name
person_name = input("Enter person name: ")

# Create folder path
dataset_path = "dataset/" + person_name

# Create folder if not exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
camera = cv2.VideoCapture(0)

count = 0  # number of images saved

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize face
        face = cv2.resize(face, (100, 100))

        # Save face image
        img_path = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(img_path, face)

        count += 1

        # Draw rectangle
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    # Stop after collecting 30 images
    if count >= 30:
        break

    # Press Q to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Face images collected successfully!")