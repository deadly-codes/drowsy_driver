from scipy.spatial import distance as dist # type: ignore
from imutils.video import VideoStream # type: ignore
from imutils import face_utils # type: ignore
from threading import Thread
import numpy as np # type: ignore
import argparse
import imutils # type: ignore
import time
import dlib # type: ignore
import cv2 # type: ignore
import os
from pygame import mixer  # type: ignore


mixer.init()
mixer.music.load("music.wav")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/sm/Desktop/nodcontrol.avi',fourcc, 20.0, (640,480))

#define movement threshodls
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 175

#find the face in the image
face_found = False
frame_num = 0

#define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
  
def distance(x,y):
    import math
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

#function to get coordinates
def get_coords(p1):
    try: return int(p1[0][0][0]), int(p1[0][0][1])
    except: return int(p1[0][0]), int(p1[0][1])

# def alarm(msg):
#     global alarm_status
#     global alarm_status2
#     global saying

#     while alarm_status:
#         print('call')
#         s = 'espeak "'+msg+'"'
#         os.system(s)

#     if alarm_status2:
#         print('call')
#         saying = True
#         s = 'espeak "' + msg + '"'
#         os.system(s)
#         saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 35
# alarm_status = False
# alarm_status2 = False
# saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
# time.sleep(1.5)

feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while not face_found:
    # Take first frame and find corners in it
    frame_num += 1
    frame = vs.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face_found = True
    # cv2.imshow('image',frame)
    out.write(frame)
    cv2.waitKey(1)
face_center = x+w/2, y+h/3
p0 = np.array([[face_center]], np.float32)

gesture = False
x_movement = 0
y_movement = 0
gesture_show = 200 #number of frames a gesture is shown


while True:

    frame = vs.read()
    old_gray = frame_gray.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    cv2.circle(frame, get_coords(p1), 4, (0,0,255), -1)
    cv2.circle(frame, get_coords(p0), 4, (255,0,0))
    
    #get the xy coordinates for points p0 and p1
    a,b = get_coords(p0), get_coords(p1)
    x_movement += abs(a[0]-b[0])
    y_movement += abs(a[1]-b[1])
    
    # text = 'x_movement: ' + str(x_movement)
    # if not gesture: cv2.putText(frame,text,(300,110), font, 0.8,(0,0,255),2)
    # text = 'y_movement: ' + str(y_movement)
    # if not gesture: cv2.putText(frame,text,(50,150), font, 0.8,(0,0,255),2)
    
    # if x_movement > gesture_threshold:
    #     gesture = False
    if x_movement > gesture_threshold or y_movement > gesture_threshold:
        gesture = True
    if gesture and gesture_show > 0:
        cv2.putText(frame, "Sit Straight", (10, 110),
                            font, 0.7, (0, 0, 255), 2)
        print("Moving Head")
        mixer.music.play()
        gesture_show -=1
    if gesture_show == 0:
        gesture = False
        x_movement = 0
        y_movement = 0
        gesture_show = 60 #number of frames a gesture is shown
        
    #print distance(get_coords(p0), get_coords(p1))
    p0 = p1
    
    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if alarm_status == False:
                #     alarm_status = True
                #     t = Thread(target=alarm, args=('wake up sir',))
                #     t.deamon = True
                #     t.start()

                cv2.putText(frame, "careful you are sleepy", (10, 30),
                            font, 0.7, (0, 0, 255), 2)
                print("Blinking")
                mixer.music.play()
        else:
            COUNTER = 0
            # alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "careful you are sleepy", (10, 50),
                            font, 0.7, (0, 0, 255), 2)
                print("Yawning")
                mixer.music.play()
                # if alarm_status2 == False and saying == False:
                #     alarm_status2 = True
                #     t = Thread(target=alarm, args=('take some fresh air sir',))
                #     t.deamon = True
                #     t.start()
                    
        # else:
        #     alarm_status2 = False

        cv2.putText(frame, "BLINK: {:.2f}".format(ear), (400, 30),
                    font, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (400, 70),
                    font, 0.7, (0, 0, 255), 2)
        
        text = 'x_movement: ' + str(x_movement)
        cv2.putText(frame,text,(350,110), font, 0.8,(0,0,255),2)
        text = 'y_movement: ' + str(y_movement)
        cv2.putText(frame,text,(350,150), font, 0.8,(0,0,255),2)


    cv2.imshow("Frame", frame)
    out.write(frame)
    if cv2.waitKey(1) and 0xFF == ord("q"):
          break

cv2.destroyAllWindows()
vs.stop()