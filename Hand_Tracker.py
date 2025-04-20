import cv2 as cv
import mediapipe as mp
import time

#Web Cam set up
web_camp = cv.VideoCapture(0)
web_camp.set(cv.CAP_PROP_FRAME_WIDTH, 640)
web_camp.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

#Media pipe set up
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    isTrue , frame = web_camp.read()
    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,i,mpHand.HAND_CONNECTIONS)

    if isTrue:
        cv.imshow('web_camp',frame)
        if cv.waitKey(20) & 0xff == ord('d'):
            break
    else:
        break

web_camp.release()
cv.destroyAllWindows()