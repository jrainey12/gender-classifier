from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import sys
from predict import main as predict
from functools import partial
import tensorflow as tf
from tensorflow import keras

class Window(QMainWindow):
    """
    UI for testing gender classifier with various gait data types.
    """
    def __init__(self):
        super(Window, self).__init__()
        
        self.setGeometry(100, 100, 1000, 760)
        self.setWindowTitle("Gender Classifier")
        self.input_type = "GEI" 
        self.model = "GEI/gender_model"

        #select type
        self.cb = QComboBox(self)
        self.cb.addItems(["GEI", "Single Frame", "Sub GEIs", "Local Frame Average", "Key Frames"])
        self.cb.setFixedSize(170,50)
        self.cb.move(707, 400)
        self.cb.currentIndexChanged.connect(self.selectionchange)
        
        self.cb_label = QLabel(self)               
        self.cb_label.setText("Input Type: ")
        self.cb_label.setFixedSize(80,50)
        self.cb_label.move(627,400)

        #Select image
        self.button = QPushButton(self)
        self.button.clicked.connect(self.file_open)
        self.button.setText("Select Image")
        self.button.setFixedSize(250,50)
        self.button.move(627,500)

        #Get prediction
        self.button = QPushButton(self)
        self.button.clicked.connect(lambda: self.get_results())
        self.button.setText("Classify")
        self.button.setFixedSize(250,50)
        self.button.move(627,600)

        # Show preview of image
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
        """
        Open an image file and set preview image
        """
        name = QFileDialog.getOpenFileName(self, 'Open File')
        self.image = name[0]
        pixmap = QtGui.QPixmap(name[0])
        self.im_preview.setPixmap(pixmap.scaled(self.im_preview.size()))


    def get_results(self):
        """
        Get the results of the prediction and update the text output.
        """
        print (self.input_type)
        results = self.classify()        
        
        text = " Predicted Gender: " + str(results[0]) + "\n Probability: " + str(results[1])
        self.results.setText(text)


    def selectionchange(self,i):
        """
        Update the input type and the model to be loaded.
        """
        self.input_type = self.cb.currentText()

        if self.input_type == "GEI":

            self.model = "GEI/gender_model"
        
        elif self.input_type == "Single Frame":

            self.model = "single_frame/gender_model"

        print ("Current index",i,"selection changed ",self.cb.currentText())



    def classify(self):
        """
        Perform prediction with the loaded the image and model.
        """
        if self.input_type == "GEI":

            prediction = predict(self.image,self.model)

        elif self.input_type == "Single Frame":

            prediction = predict(self.image,self.model)
    
        else:
            print("Input Type not implemented yet.")
            prediction = ["None", 0]
        return prediction
 


if __name__=='__main__':

    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())
