import sqlite3
import sys

import numpy
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time





class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('Attendance_module_main.ui', self)

        self.submit_button.clicked.connect(self.TrackImages)
        self.VBL = QVBoxLayout()
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.start()

    def ImageUpdateSlot(self, Image, frame):
        self.Feed.setPixmap(QPixmap.fromImage(Image))
        self.npnda=frame

    npnda=numpy.ndarray




    def TrackImages(self):

        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        recognizer.read("TrainingImageLabel\Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
        #cam = cv2.VideoCapture('http://192.168.1.100:8080/video')
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        #while self.fg==True:
            #ret, im = cam.read()
        im=self.npnda
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                #aa = df.loc[df['Id'] == Id]['Name'].values
                db = sqlite3.connect('db_attendance.db')
                c = db.cursor()
                tm = c.execute("SELECT NAME FROM tb_test WHERE ID = (?)", (Id,))
                aa = list(tm)[0][0]



                tt = str(Id)
                attendance.loc[len(attendance)] = [Id, date, timeStamp]
                today = datetime.date.today()
                d2 = today.strftime("%B %d, %Y")
                t = time.localtime()
                current_time = time.strftime("%I:%M %p", t)
                c.execute("INSERT INTO tb_attendance_data (ID, DATETIME) values(?,CURRENT_TIMESTAMP)",(Id,))

                db.commit()

                self.time_dis.setText(current_time)
                self.date_dis.setText(d2)
                #name_ret = aa.item(0)
                self.name_dis.setText(aa)

                self.id_dis.setText(str(Id))
                self.Status.setText("Attendance Marked!")
                QMessageBox.about(self, "Completed",
                                  "Congratulations ! Your attendance has been marked successfully for the day!!")

            else:
                Id = 'Unknown'
                tt = str(Id)
                QMessageBox.about(self, "Error",
                                  "Face not Recognized")
                self.Status.setText("Error, Try again!")
            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) +
                            ".jpg", im[y:y + h, x:x + w])
                cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            #cv2.imshow('im', im)
            #if (cv2.waitKey(1) == ord('q')):
            #    break
            #break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%h:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
        attendance.to_csv(fileName, index=False)
        #cam.release()
        #cv2.destroyAllWindows()
        res = "Attendance Taken"
        #self.attendance_output_label.setText(res)
        #QMessageBox.about(self, "Completed", "Congratulations ! Your attendance has been marked successfully for the day!!")




class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage, numpy.ndarray)

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture("http://192.168.1.100:8080/video")
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        recognizer.read("TrainingImageLabel\Trainner.yml")
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        while self.ThreadActive:
            ret, frame = Capture.read()
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if ret:

                for (x, y, w, h) in faces:

                    cv2.rectangle(Image, (x, y), (x + w, y + h), (225, 0, 0), 2)
                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    if (conf < 50):
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(
                            ts).strftime('%H:%M:%S')
                        #aa = df.loc[df['Id'] == Id]['Name'].values
                        #tt = str(Id) + "-" + aa
                        db = sqlite3.connect('db_attendance.db')
                        c = db.cursor()
                        tm = c.execute("SELECT NAME FROM tb_test WHERE ID = (?)", (Id,))
                        aa=list(tm)[0][0]

                        tt = str(Id) + "-" + str(aa)
                        attendance.loc[len(attendance)] = [Id, date, timeStamp]

                    else:
                        Id = 'Unknown'
                        tt = str(Id)
                    if (conf > 75):
                        noOfFile = len(os.listdir("ImagesUnknown")) + 1
                        cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) +
                                    ".jpg", Image[y:y + h, x:x + w])

                    cv2.putText(Image, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
                    cv2.rectangle(Image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(500, 282, Qt.KeepAspectRatio)
                pixmap_flipped = Pic.transformed(QTransform().scale(-1, 1))
                self.ImageUpdate.emit(pixmap_flipped, frame)

    def stop(self):
        self.ThreadActive = False
        self.quit()


app=QApplication(sys.argv)
mainwindow=Window()
mainwindow.show()
app.exec_()
