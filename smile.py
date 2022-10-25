#smile detection - webcam
import cv2
import numpy as np

vid = cv2.VideoCapture(0)
smile_cascade = cv2.CascadeClassifier('smile.xml')

while True:
    ret,frame = vid.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    smiles = smile_cascade.detectMultiScale(gray,1.5,7)

    for (x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('vid',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):

        break

vid.release()
cv2.destroyAllWindows()
