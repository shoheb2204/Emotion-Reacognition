import numpy as np
import cv2
from fer import Video
from fer import FER
import os
import sys
import pandas as pd

cap = cv2.VideoCapture(0)
emo_detector = FER(mtcnn=True)

while 1:
    ret, img = cap.read()
    #captured_emotions = emo_detector.detect_emotions(img)
    dominant_emotion, emotion_score = emo_detector.top_emotion(img)

    #if len(captured_emotions)>0:
    if dominant_emotion!=None:
        #captured_e=captured_emotions[0]
        #Cordinate=captured_e['box']
        #x=Cordinate[0]
        #y=Cordinate[1]
        #w=Cordinate[2]
        #h=Cordinate[3]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        pos=(30,30)
        fontScale=1
        fontColor=(255,0,0)
        lineType=2
                
        cv2.putText(img,str(dominant_emotion),pos,font,fontScale,fontColor,lineType)
         

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
