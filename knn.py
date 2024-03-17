from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
import csv


video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('D:/Users/Aarad/PycharmProjects/FaceCheck/haarcascade_frontalface_default.xml')

with open('D:/Users/Aarad/PycharmProjects/FaceCheck/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('D:/Users/Aarad/PycharmProjects/FaceCheck/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

#print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

#imgBackground=cv2.imread("background.png")

COL_NAMES = ['Student', 'Time']
#attendance = []
attendance_taken = ""
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
exist=os.path.isfile("D:/Users/Aarad/PycharmProjects/FaceCheck/rollcall_" + date + ".csv")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist=os.path.isfile("D:/Users/Aarad/PycharmProjects/FaceCheck/rollcall_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance = [str(output[0]), str(timestamp)]
        #if attendance_taken != str(output[0]):
         #   with open("D:/Users/Aarad/PycharmProjects/FaceCheck/rollcall_" + date + ".csv", "+a") as csvfile:
          #      writer = csv.writer(csvfile)
           #     writer.writerow(attendance)
            #csvfile.close()
            #attendance_taken = attendance
            #attendance = []
    #imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k == ord('h'):
        if exist:
            with open("D:/Users/Aarad/PycharmProjects/FaceCheck/rollcall_" + date + ".csv", "+a")as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
                print(output)
            csvfile.close()
        else:
            with open("D:/Users/Aarad/PycharmProjects/FaceCheck/rollcall_" + date + ".csv", "+a")as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                #print(attendance)
            csvfile.close()
    if k==ord('q'):
        break

       # if attendance_taken[0] == attendance[0]:
        #    attendance = []
        #attendance_taken = attendance

video.release()
cv2.destroyAllWindows()
