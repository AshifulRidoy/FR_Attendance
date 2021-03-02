import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QMessageBox
from PyQt5.uic import loadUi
import sqlite3
import cv2
import numpy as np
import pandas as pd
import csv
import os
from PIL import Image

db = sqlite3.connect('db_attendance.db')
c = db.cursor()


class Admin_Module(QMainWindow):
    def __init__(self):
        super(Admin_Module, self).__init__()
        loadUi("admin_module.ui", self)

        self.registered_student_button.clicked.connect(
            self.registered_student_function)

        self.report_button.clicked.connect(
            self.report_function)

        self.capture_image_button.clicked.connect(
            self.capture_image_function)

        self.train_image_button.clicked.connect(
            self.train_image_function)

    def registered_student_function(self):
        RegisteredStudents = RegisteredStudentsTable()
        RegisteredStudents.exec_()

    def report_function(self):
        Report = report()
        Report.exec_()

    def capture_image_function(self):
        CaptureImage = Capture_Image()
        CaptureImage.exec_()

    def train_image_function(self):
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        faces, Id = self.getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save("TrainingImageLabel\Trainner.yml")

        QMessageBox.about(self, 'Completed',
                          'Your model has been trained successfully!!')

    def getImagesAndLabels(self, path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        Ids = []

        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            #Id = int(os.path.split(imagePath)[-1].split(".")[1])
            Id = int(os.path.split(imagePath)[-1].split(".")[0])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids


class RegisteredStudentsTable(QDialog):
    def __init__(self):
        super(RegisteredStudentsTable, self).__init__()
        loadUi('registered_students_table.ui', self)
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 300)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 300)
        self.loaddata()

    def loaddata(self):
       # sql = 'SELECT * FROM tb_student '
        c = db.execute("SELECT * FROM tb_student")

       # results = c.execute(sql)

        self.tableWidget.setRowCount(0)
        for row_number, row_data in enumerate(c):
            self.tableWidget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.tableWidget.setItem(
                    row_number, column_number, QtWidgets.QTableWidgetItem(str(data)))


class report(QDialog):
    def __init__(self):
        super(report, self).__init__()
        loadUi('report.ui', self)
        self.pushButton.clicked.connect(self.printdate)

    def printdate(self):

        temp = self.dateEdit.date()
        temp2 = temp.toPyDate()
        temp2.strftime('%m/%d/%Y')
        self.label.setText(temp2.strftime('%m/%d/%Y'))


class Capture_Image(QDialog):
    def __init__(self):
        super(Capture_Image, self).__init__()
        loadUi('Capture_Image.ui', self)
        self.start_btn.clicked.connect(
            self.TakeImages)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def TakeImages(self):
        Id = self.id_input.text()

        if not Id:
            res = "Please enter Id"
            self.status_label.setText(res)
            QMessageBox.about(
                self, "Error", "Please enter roll number properly")

        elif (self.is_number(Id)):
            cam = cv2.VideoCapture('http://192.168.1.100:8080/video')
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ " + Id + '.' +
                                str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # display the frame
                    cv2.imshow('frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 60:
                    break
            cam.release()
            cv2.destroyAllWindows()
            res = "Images Saved for ID : " + Id
           # row = [Id]
            # with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            ##  writer = csv.writer(csvFile)
            # writer.writerow(row)
          #  csvFile.close()

            c.execute("""INSERT INTO tb_test (ID) VALUES (?)""", (Id,))
            db.commit()
            self.status_label.setText(res)
        else:

            if (Id.isalpha()):
                res = "Enter Numeric Id"
                self.status_label.setText(res)


# main
app = QApplication(sys.argv)
mainwindow = Admin_Module()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(400)
widget.setFixedHeight(480)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
