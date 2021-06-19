import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personName = []
mylist = os.listdir(path)
print(mylist)
for cu_img in mylist:
    current_img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])

print(personName)

def faceEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknow = faceEncodings(images)
print("All Encoding Complete!!!")

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myData = f.readline()
        nameList = []
        for line in myData:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tstr},{dstr}\n')


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrent = face_recognition.face_locations(faces)
    encodesCurrentframe = face_recognition.face_encodings(faces, facesCurrent)

    for encodeFace, faceloc in zip(encodesCurrentframe, facesCurrent):
        matches = face_recognition.compare_faces(encodelistknow, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknow, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            string = "Attendence marked."
            # print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2-20), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, string, (x1 - 100, y2 + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            attendance(name)

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()
