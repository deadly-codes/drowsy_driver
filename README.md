# IS_DRIVER_DROWSY

## About The Project :

This project helps you detect that if the driver is drunk / sleepy / drowsy amd if thats True then it'll beep alarm to alert you.
This project increases safety on the roads: Drowsy driving is a major contributor to accidents, and a drowsy driver detection system could help prevent these accidents by alerting drivers when they become too fatigued to drive safely.

## About files :

- "drowsiness.py" - has full working model
- "eye_blinking.py" - detects eye blinking
- "yawn.py" - detects the yawning
- "head.py" - detcts head correct position
- "music.wav" - it is the beep alarm
- "dlib-19.24.1-cp311-cp311-win_amd64.whl" - it is the package/file to install dlib library
- "shape_predictor_68_face_landmarks.dat" - pre-data file present below
- "haarcascade_frontalface_alt.xml" and "haarcascade_frontalface_alt.xml" - an XML file containing serialized Haar cascade detector of faces in the OpenCV library.

## Programming Languages- Python

Python version (Recommended) - 2.x or 3.x

## Dependencies - cmake, dlib, Opencv, imutils, scipy, pygame

CAN INSTALL SOME PYTHON PACKAGES BY :

- `pip install cmake`
- `pip install "dlib-19.24.1-cp311-cp311-win_amd64.whl"` (file present above)
- `pip install opencv-python`
- `pip install imutils`
- `pip install scipy`
- `pip install pygame`

## About Pre-Data : 
File contains 68 landmarks, when the model will detect landmarks from our face (these landmarks are some points on our face which includes eyes, eyebrows, jawline ,lips,etc), it'll match to the pre-data for analysing.

Dat file is here - [CLICK ME !](https://drive.google.com/file/d/1vGljjJ2l4tjhiOHpqyS_R-_ihUKRLvE4/view?usp=sharing) .

## IDE - Python IDLE or VSCODE
