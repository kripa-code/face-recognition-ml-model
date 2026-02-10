
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows


python3 -m pip install --upgrade pip

python3 -m pip install opencv-python
python3 -m pip install scikit-learn
python3 -m pip install numpy

python3 collect_faces.py
python3 train_model.py
python3 recognize_face.py
