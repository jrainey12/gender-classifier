from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import sys
from predict import main as predict
from functools import partial


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        
        self.setGeometry(100, 100, 1000, 760)
        self.setWindowTitle("Gender Classifier")
        self.input_type = "GEI" 

        self.cb = QComboBox(self)
        self.cb.addItems(["GEI", "Single Frame", "Sub GEIs", "Local Frame Average", "Key Frames"])
        self.cb.setFixedSize(170,50)
        self.cb.move(707, 400)
        self.cb.currentIndexChanged.connect(self.selectionchange)
        
        self.cb_label = QLabel(self)               
        self.cb_label.setText("Input Type: ")
        self.cb_label.setFixedSize(80,50)
        self.cb_label.move(627,400)

        self.button = QPushButton(self)
        self.button.clicked.connect(self.file_open)
        self.button.setText("Select Image")
        self.button.setFixedSize(250,50)
        self.button.move(627,500)

        self.button = QPushButton(self)
        self.button.clicked.connect(lambda: self.get_results(self.input_type))
        self.button.setText("Classify")
        self.button.setFixedSize(250,50)
        self.button.move(627,600)


       # openFile = QAction("&File", self)
       # openFile.setShortcut("Ctrl+O")
       # openFile.setStatusTip("Open File")
       # openFile.triggered.connect(self.file_open)

        #self.statusBar()

        #mainMenu = self.menuBar()

        #fileMenu = mainMenu.addMenu('&File')
        #fileMenu.addAction(openFile)

        self.im_preview = QLabel(self)
        self.im_preview.setText("Image Preview")
        self.im_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.im_preview.setGeometry(35,60,440,640)
        self.im_preview.setStyleSheet("border: 1px solid black;") 

        self.results_title = QLabel(self)
        self.results_title.setText("Classification Results")
        self.results_title.setAlignment(QtCore.Qt.AlignHCenter)
        self.results_title.setGeometry(515,60,440,20)
        self.results_title.setStyleSheet("border: 1px solid black;") 

        self.results = QLabel(self)
        self.results.setText("Press \"Select Image\" to choose a image for classification \nPress \"Classify\" to get a prediction.")
        self.results.setAlignment(QtCore.Qt.AlignCenter)
        self.results.setGeometry(515,80,440,300)
        self.results.setStyleSheet("border: 1px solid black;background-color: white;color: black;") 

        

        self.show()

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        global image 
        image = name[0]
       # print(image)
        pixmap = QtGui.QPixmap(name[0])
        self.im_preview.setPixmap(pixmap.scaled(self.im_preview.size()))


    def get_results(self,input_type):
        print (input_type)
        results = classify(input_type)        
        
        text = " Predicted Gender: " + str(results[0]) + "\n Probability: " + str(results[1])
        print ("Setting text")    
        self.results.setText(text)


    def selectionchange(self,i):

      self.input_type = self.cb.currentText()

      print ("Current index",i,"selection changed ",self.cb.currentText())




def classify(input_type):
    global image
    print(image)
    if input_type == "GEI":

        prediction = predict(image, "GEI/gender_model")

    elif input_type == "Single Frame":

        prediction = predict(image, "single_frame/gender_model_10_epochs")
    else:
        print("Input Type not implemented yet.")
        prediction = ["None", 0]
    return prediction
 

if __name__=='__main__':

    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())
