from PyQt6.QtWidgets import (
    QApplication, 
    QWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QListWidget,
    QCheckBox,
    QSpinBox,
    QButtonGroup,
    QRadioButton,
)

from PyQt6.QtCore import *
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import requests
import sys

import Spam_Predictor_SingleData
import Spam_Predictor_MultipleData

isOpenExternal = False
modelData = Spam_Predictor_SingleData

class HeatMapWindow(QWidget):
    def __init__(self):
        super().__init__()
        global modelData
        layout = QVBoxLayout()
        self.label = QLabel("Heat Map Comparison")
        layout.addWidget(self.label)

        fig = modelData.plot_confusion_matrices(modelData.cm1, modelData.cm2, modelData.cm3, 
                            "Multinomial NB", "Gaussian NB", "Complement NB", False)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        self.setLayout(layout)
    
    def closeEvent(self, event):
        global isOpenExternal
        isOpenExternal = False
        event.accept()

class RadialWindow(QWidget):
    def __init__(self):
        super().__init__()
        global modelData
        layout = QVBoxLayout()
        self.label = QLabel("Radial Comparison")
        layout.addWidget(self.label)

        predictionList = [modelData.predictions1, modelData.predictions2, modelData.predictions3]
        fig = modelData.plot_classification_report_radial_combined(predictionList, 
                            ["Multinomial NB", "Gaussian NB", "Complement NB"])
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        self.setLayout(layout)

    def closeEvent(self, event):
        global isOpenExternal
        isOpenExternal = False
        event.accept()

class AccuracyWindow(QWidget):
    def __init__(self):
        super().__init__()
        global modelData
        layout = QVBoxLayout()
        self.label = QLabel("Accuracy Comparison")
        layout.addWidget(self.label)

        fig = modelData.plot_accuracy_comparison(modelData.accuracy1, modelData.accuracy2, modelData.accuracy3, False)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        self.setLayout(layout)

    def closeEvent(self, event):
        global isOpenExternal
        isOpenExternal = False
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        global isOpenExternal
        self.external = None

        self.modelData = Spam_Predictor_SingleData
        self.modelMNB = True
        self.modelGNB = False
        self.modelCNB = False
        self.setWindowTitle("YouTube Spam Checker")
        self.resize(1500, 2100)

        #########################################################################
        # Main Layout
        self.layout = QVBoxLayout()
        self.gridLayoutDataOption = QGridLayout()
        self.gridLayoutData = QGridLayout()
        self.gridLayoutModel = QGridLayout()

        # Create the menu bar
        menu_bar = self.menuBar()

        # View menu
        file_menu = menu_bar.addMenu("View")
        
        # View -> HeatMap
        hmAction = QAction("HeatMap", self)
        hmAction.triggered.connect(self.heatMap)
        file_menu.addAction(hmAction)

        # View -> Radial
        rAction = QAction("Radial", self)
        rAction.triggered.connect(self.radialMap)
        file_menu.addAction(rAction)

         # View -> Accuracy
        aAction = QAction("Accuracy", self)
        aAction.triggered.connect(self.accuracyMap)
        file_menu.addAction(aAction)

        #########################################################################
        # First Part - Youtube
        self.titleLabel = QLabel("YouTube ID")
        self.videoIDTextBox = QLineEdit()

        self.commentLabel = QLabel("Max Comment Number (Max 5000)")
        self.commentNoSpinBox = QSpinBox()
        self.commentNoSpinBox.setMaximum(5000)

        self.analyzeButton = QPushButton("Analyze Comments")
        self.analyzeButton.clicked.connect(self.analyze_comments)

        self.clearDataButton = QPushButton("Clear Data")
        self.clearDataButton.clicked.connect(self.clear_data)

        self.modelLabel = QLabel("Data")
        self.radioBtnGrp = QButtonGroup()
        self.modelRadioBtn1 = QRadioButton("Default")
        self.modelRadioBtn2 = QRadioButton("Enhanced")
        self.modelRadioBtn1.setChecked(True)

        self.radioBtnGrp.addButton(self.modelRadioBtn1)
        self.radioBtnGrp.addButton(self.modelRadioBtn2)

        self.gridLayoutDataOption.addWidget(self.modelLabel, 0, 0)
        self.gridLayoutDataOption.addWidget(self.modelRadioBtn1, 1, 0)
        self.gridLayoutDataOption.addWidget(self.modelRadioBtn2, 1, 1)

        self.modelRadioBtn1.toggled.connect(self.update_data)
        self.modelRadioBtn2.toggled.connect(self.update_data)

        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.videoIDTextBox)
        self.layout.addWidget(self.commentLabel)
        self.layout.addWidget(self.commentNoSpinBox)
        self.layout.addLayout(self.gridLayoutDataOption)
        self.layout.addWidget(self.analyzeButton)
        self.layout.addWidget(self.clearDataButton)

        #########################################################################
        # Second Part - Model Option
        self.modelLabel = QLabel("Model")
        self.modelCheckBtn1 = QCheckBox("Multinomial Naive Bayes")
        self.modelCheckBtn2 = QCheckBox("Gaussian Naive Bayes")
        self.modelCheckBtn3 = QCheckBox("Complement Naive Bayes")
        self.modelCheckBtn1.setChecked(True)

        self.gridLayoutModel.addWidget(self.modelLabel, 0, 0)
        self.gridLayoutModel.addWidget(self.modelCheckBtn1, 1, 0)
        self.gridLayoutModel.addWidget(self.modelCheckBtn2, 1, 1)
        self.gridLayoutModel.addWidget(self.modelCheckBtn3, 1, 2)

        self.modelCheckBtn1.stateChanged.connect(self.update_model)
        self.modelCheckBtn2.stateChanged.connect(self.update_model)
        self.modelCheckBtn3.stateChanged.connect(self.update_model)

        self.layout.addLayout(self.gridLayoutModel)

        #########################################################################
        # Third Part - Data output
        self.dataLabel = QLabel("Multinomial Naive Bayes")
        self.spam = QListWidget()
        self.nonSpam = QListWidget()
        self.spamLabel = QLabel("Spam")
        self.nonSpamLabel = QLabel("Non-Spam")

        self.gridLayoutData.addWidget(self.dataLabel, 0, 0)
        self.gridLayoutData.addWidget(self.spamLabel, 1, 0)
        self.gridLayoutData.addWidget(self.nonSpamLabel, 3, 0)
        self.gridLayoutData.addWidget(self.spam, 2, 0)
        self.gridLayoutData.addWidget(self.nonSpam, 4,0)

        self.layout.addLayout(self.gridLayoutData)

        self.data2Label = QLabel("Gaussian Naive Bayes")
        self.spam2 = QListWidget()
        self.nonSpam2 = QListWidget()
        self.spam2Label = QLabel("Spam")
        self.nonSpam2Label = QLabel("Non-Spam")

        self.data3Label = QLabel("Complement Naive Bayes")
        self.spam3 = QListWidget()
        self.nonSpam3 = QListWidget()
        self.spam3Label = QLabel("Spam")
        self.nonSpam3Label = QLabel("Non-Spam")
        
        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.setCentralWidget(self.widget)

        self.setStyleSheet(self.get_styles())
     

    def get_styles(self):
        return """
        QMainWindow {
            background-color: #121212;
        }
        QMenuBar {
            color: #E0E0E0;
        }
        QMenur {
            color: #E0E0E0;
        }
        QLabel {
            color: #E0E0E0;
            font-size: 14px;
        }
        QLineEdit, QSpinBox {
            background-color: #333333;
            color: #FFFFFF;
            border: 1px solid #555555;
            padding: 4px;
        }
        QPushButton {
            background-color: #444444;
            color: #FFFFFF;
            border: 1px solid #555555;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QCheckBox {
            color: #E0E0E0;
        }
        QRadioButton {
            color: #E0E0E0;
        }
        QListWidget {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: 1px solid #555555;
        }
        """

    def heatMap(self, checked):
        global isOpenExternal

        if isOpenExternal == False:
            self.external = None

        if self.external is None:
            isOpenExternal = True
            self.external = HeatMapWindow()
            self.external.show()
        else:
            isOpenExternal = False
            self.external = None
    
    def radialMap(self, checked):
        global isOpenExternal
        if isOpenExternal == False:
            self.external = None

        if self.external is None:
            isOpenExternal = True
            self.external = RadialWindow()
            self.external.show()
        else:
            self.external = None
    
    def accuracyMap(self, checked):
        global isOpenExternal
        if isOpenExternal == False:
            self.external = None

        if self.external is None:
            isOpenExternal = True
            self.external = AccuracyWindow()
            self.external.show()
        else:
            isOpenExternal = False
            self.external = None

    def clear_data(self):
        self.spam.clear()
        self.nonSpam.clear()
        self.spam2.clear()
        self.nonSpam2.clear()
        self.spam3.clear()
        self.nonSpam3.clear()
    
    def update_data(self,):
        global modelData

        if self.modelRadioBtn1.isChecked():
            modelData = Spam_Predictor_SingleData
        elif self.modelRadioBtn2.isChecked():
            modelData = Spam_Predictor_MultipleData
    def update_model(self, state):

        noOfChecked = 0

        if not self.modelCheckBtn1.isChecked() and not self.modelCheckBtn2.isChecked() and not self.modelCheckBtn3.isChecked():
            sender = self.sender()
            sender.setChecked(True)

        if self.modelCheckBtn1.isChecked():
            self.gridLayoutData.addWidget(self.dataLabel, 0, noOfChecked)
            self.gridLayoutData.addWidget(self.spamLabel, 1, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpamLabel, 3, noOfChecked)
            self.gridLayoutData.addWidget(self.spam, 2, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpam, 4,noOfChecked)
            noOfChecked += 1
            self.modelMNB = True
        else:
            self.dataLabel.setParent(None)
            self.spamLabel.setParent(None)
            self.nonSpamLabel.setParent(None)
            self.spam.setParent(None)
            self.nonSpam.setParent(None)
            self.modelMNB = False

        if self.modelCheckBtn2.isChecked():
            self.gridLayoutData.addWidget(self.data2Label, 0, noOfChecked)
            self.gridLayoutData.addWidget(self.spam2Label, 1, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpam2Label, 3, noOfChecked)
            self.gridLayoutData.addWidget(self.spam2, 2, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpam2, 4, noOfChecked)
            noOfChecked += 1
            self.modelGNB = True
        else:
            self.data2Label.setParent(None)
            self.spam2Label.setParent(None)
            self.nonSpam2Label.setParent(None)
            self.spam2.setParent(None)
            self.nonSpam2.setParent(None)
            self.modelGNB = False

        if self.modelCheckBtn3.isChecked():
            self.gridLayoutData.addWidget(self.data3Label, 0, noOfChecked)
            self.gridLayoutData.addWidget(self.spam3Label, 1, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpam3Label, 3, noOfChecked)
            self.gridLayoutData.addWidget(self.spam3, 2, noOfChecked)
            self.gridLayoutData.addWidget(self.nonSpam3, 4, noOfChecked)
            noOfChecked += 1
            self.modelCNB = True
        else:
            self.data3Label.setParent(None)
            self.spam3Label.setParent(None)
            self.nonSpam3Label.setParent(None)
            self.spam3.setParent(None)
            self.nonSpam3.setParent(None)
            self.modelCNB = False

    def analyze_comments(self):
        # Sample video ID: mzwlld2lbko
        # LMFAO Video: KQ6zr6kCPj8, wyx6JDQCslE
        global modelData

        noOfComments = self.commentNoSpinBox.value()
        if noOfComments >= 100 :
            totalComment = '100'
        else:
            totalComment = str(noOfComments)
        spamCount = 1
        nonSpamCount = 1
        spam2Count = 1
        nonSpam2Count = 1
        spam3Count = 1
        nonSpam3Count = 1
        key = "AIzaSyBZ8PC5GtUdn1g78eYh_mh0LMr_iNozYgI"
        url = "https://www.googleapis.com/youtube/v3/commentThreads?key=" + key + "&textFormat=plainText&part=snippet&videoId=" + self.videoIDTextBox.text() + "&maxResults=" + totalComment
        
        
        while noOfComments > 0:
            if self.videoIDTextBox.text() != "":

                try:
                    response = requests.get(url)
                    if  response.status_code == 200:

                        # Loop thru each comment
                        data = response.json()
                        for item in data['items']:
                            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                            
                            if self.modelMNB == True:

                                prediction1 = modelData.predictor([comment], 1)

                                if prediction1[0] == 0:
                                    self.nonSpam.addItem(str(spamCount) + ":  " + comment)
                                    spamCount += 1
                                elif prediction1[0] == 1:
                                    self.spam.addItem(str(nonSpamCount) + ":  " + comment)
                                    nonSpamCount += 1
                            
                            if self.modelGNB == True:

                                prediction2 = modelData.predictor([comment], 2)

                                if prediction2[0] == 0:
                                    self.nonSpam2.addItem(str(spam2Count) + ":  " + comment)
                                    spam2Count += 1
                                elif prediction2[0] == 1:
                                    self.spam2.addItem(str(nonSpam2Count) + ":  " + comment)
                                    nonSpam2Count += 1

                            if self.modelCNB == True:

                                prediction3 = modelData.predictor([comment], 3)

                                if prediction3[0] == 0:
                                    self.nonSpam3.addItem(str(spam3Count) + ":  " + comment)
                                    spam3Count += 1
                                elif prediction3[0] == 1:
                                    self.spam3.addItem(str(nonSpam3Count) + ":  " + comment)
                                    nonSpam3Count += 1

                        noOfComments -= 100
                        if noOfComments < 100:
                            totalComment = str(noOfComments)
                        else:
                            totalComment = "100"
                        url = "https://www.googleapis.com/youtube/v3/commentThreads?key=" + key + "&textFormat=plainText&part=snippet&videoId=" + self.videoIDTextBox.text() + "&maxResults=" + totalComment
                        url = url + "&pageToken=" + data['nextPageToken']

                    else:
                        print('Error: ', response.status_code)
                        print('Error: ', response.json())
                        break
                except requests.exceptions.RequestException as e:
                    print('Error:', e)
            else:
                print("Please enter a valid Youtube ID")
    

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
