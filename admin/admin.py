import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi


import sqlite3

db = sqlite3.connect('db_attendance.db')
c = db.cursor()


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi("login.ui", self)
        self.loginbutton.clicked.connect(self.loginfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)

    def loginfunction(self):
        try:
            username = self.username.text()
            password = self.password.text()

            c.execute("SELECT username,password from tb_admin where username like '" +
                      username + "'and password like '"+password+"'")
            result = c.fetchone()

            if result == None:
                self.labelResult.setText("Incorrect Email & Password")

            else:
                self.labelResult.setText("You are logged in")
                import admin_main

        except sqlite3.Error as e:
            self.labelResult.setText("Error")


app = QApplication(sys.argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(400)
widget.setFixedHeight(480)
widget.show()
app.exec_()
