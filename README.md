# Face Recognition ML Model

Simple face dataset collection, training, and recognition pipeline using OpenCV.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Collect faces:

```bash
python collect_faces.py
```

Train the model:

```bash
python train_model.py
```

Recognize faces:

```bash
python recognize_face.py
```

## Notes
- The dataset is stored in `dataset/`.
- The trained model file `face_recognition_model_3.pkl` is included.
