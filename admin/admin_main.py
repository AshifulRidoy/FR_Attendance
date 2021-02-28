import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication
from PyQt5.uic import loadUi


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


app = QApplication(sys.argv)
mainwindow = Admin_Module()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(680)
widget.setFixedHeight(620)
widget.show()
app.exec_()
