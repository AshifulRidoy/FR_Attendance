import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication
from PyQt5.uic import loadUi
import sqlite3

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

    def registered_student_function(self):
        RegisteredStudents = RegisteredStudentsTable()
        RegisteredStudents.exec_()

    def report_function(self):
        Report = report()
        Report.exec_()


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
