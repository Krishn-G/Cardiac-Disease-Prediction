import os
import pickle
import sys
import webbrowser

import pandas as pd
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, pyqtSlot, QAbstractTableModel, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from results import Ui_Dialog_Accuracy
from sklearn.preprocessing import StandardScaler, Normalizer

import icon_resource
# pyrcc5 resources.qrc -o resources.py


class CardioVascular_Disease_Prediction(QMainWindow):

    def __init__(self):
        super(CardioVascular_Disease_Prediction, self).__init__()
        self.df_Test_Case_final = None
        self.df_Test_Case = None
        self.splashscreen = None
        self.pix = None
        loadUi('MainWindow_Gui.ui', self)
        os.system('cls')

        self.selected_algo = str(self.Algo_Selector_ComboBox.currentText())
        print(self.selected_algo)

        self.qm = QMessageBox()

        self.AccuracyButton.clicked.connect(self.AccuracyButtonSlotFunction)
        self.confirm_manual_data_pushButton.clicked.connect(self.Create_Manual_Test_Data)
        self.confirm_patient_pushButton.clicked.connect(self.Create_CSV_Test_Data)

        self.BrowseButton.clicked.connect(self.BrowseButtonSlotFunction)
        self.DetectionButton.clicked.connect(self.DetectionButtonSlotFunction)
        self.ExitButton.clicked.connect(self.ExitButtonSlotFunction)

        self.data_array = [[]]
        self.data_header = []

        self.showSplashScreen()

    def showSplashScreen(self):
        self.pix = QPixmap('./images/splashscreen.png')
        self.splashscreen = QSplashScreen(self.pix, Qt.WindowStaysOnTopHint)
        self.splashscreen.show()
        QTimer.singleShot(2000, self.splashscreen.close)

    @pyqtSlot()
    def AccuracyButtonSlotFunction(self):

        with open('./results/KNNClassifier_Accuracy.pkl', 'rb') as f1:
            KNN_Accuracy = pickle.load(f1)
            KNN_Accuracy = round(KNN_Accuracy * 100, 2)

        with open('./results/SVMclassifier_Accuracy.pkl', 'rb') as f2:
            SVM_Accuracy = pickle.load(f2)
            SVM_Accuracy = round(SVM_Accuracy * 100, 2)

        with open('./results/GNBclassifier_Accuracy.pkl', 'rb') as f3:
            NB_Accuracy = pickle.load(f3)
            NB_Accuracy = round(NB_Accuracy * 100, 2)

        with open('./results/DTCclassifier_Accuracy.pkl', 'rb') as f4:
            DT_accuracy = pickle.load(f4)
            DT_accuracy = round(DT_accuracy * 100, 2)

        with open('./results/LRclassifier_Accuracy.pkl', 'rb') as f5:
            LogisticRegression_Accuracy = pickle.load(f5)
            LogisticRegression_Accuracy = round(LogisticRegression_Accuracy * 100, 2)

        with open('./results/RFclassifier_Accuracy.pkl', 'rb') as f8:
            RF_accuracy = pickle.load(f8)
            RF_accuracy = round(RF_accuracy * 100, 2)

        with open('./results/GBclassifier_Accuracy.pkl', 'rb') as f8:
            GB_accuracy = pickle.load(f8)
            GB_accuracy = round(GB_accuracy * 100, 2)

        datapd = {'Algorithms': ['KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'Gradient Boosting'],
                  'Accuracy': [KNN_Accuracy, SVM_Accuracy, NB_Accuracy, DT_accuracy, LogisticRegression_Accuracy, RF_accuracy, GB_accuracy]}

        df = pd.DataFrame(datapd)

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog_Accuracy()
        ui.setupUi(Dialog)
        model = pandasModel(df)
        ui.tableView.setModel(model)
        Dialog.show()
        response = Dialog.exec_()

        if response == QtWidgets.QDialog.Accepted:
            print(df)
            print('')
        else:
            print("No variable set")

        algo_names = ['KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'Gradient Boosting']
        all_acc = [KNN_Accuracy, SVM_Accuracy, NB_Accuracy, DT_accuracy, LogisticRegression_Accuracy, RF_accuracy, GB_accuracy]

        max_acc = max(all_acc)
        max_acc_index = all_acc.index(max_acc)
        best_algo_name = algo_names[max_acc_index]

        msg_text = 'The Algorithm having best accuracy is: ' + best_algo_name

        self.qm.information(self, 'Best Accuracy', msg_text, self.qm.Ok | self.qm.Ok)

    @pyqtSlot()
    def Create_Manual_Test_Data(self):
        
        age = self.age_spinBox.value()
        sex = self.gender_comboBox.currentText()
        cp = self.cp_comboBox.currentText()
        thal = self.thal_comboBox.currentText()
        slope = self.slope_comboBox.currentText()
        trestbps = self.trestbps_spinBox.value()
        chol = self.cholestrol_spinBox.value()
        fbs = self.glucose_comboBox.currentText()
        restecg = self.restecg_comboBox.currentText()
        thalach = self.thalach_spinBox.value()
        exang = self.exang_comboBox.currentText()
        oldpeak = self.oldpeak_doubleSpinBox.value()
        ca = self.ca_comboBox.currentText()  # (Number of Major Vessels)

        if sex == 'Male':
            sex_int = 0
        else:
            sex_int = 1

        thal_int = 0
        if thal == 'normal':
            thal_int = 0
        elif thal == 'fixed defect':
            thal_int = 1
        elif thal == 'reversable defect':
            thal_int = 2

        self.data_array = [[age, sex_int, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal_int]]
        self.data_header = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',  'thal']

        self.df_Test_Case = pd.DataFrame(self.data_array, columns=self.data_header)
        model = pandasModel(self.df_Test_Case)
        self.patient_data_tableView.setModel(model)

    @pyqtSlot()
    # def Create_CSV_Test_Data(self):
    #
    #     indexes = self.tableView.selectionModel().selectedRows()
    #
    #     seletedROW = indexes[0].row()
    #     print(seletedROW)
    #
    #     X_test_CSV = self.CSV_df.iloc[[seletedROW]]
    #     print(X_test_CSV)
    #     self.df_Test_Case = X_test_CSV.drop(['target'], axis=1)
    #
    #     model = pandasModel(self.df_Test_Case)
    #     self.patient_data_tableView.setModel(model)
    @pyqtSlot()
    def Create_CSV_Test_Data(self):
        try:
            indexes = self.tableView.selectionModel().selectedRows()
            if not indexes:
                self.qm.warning(self, 'No Row Selected', 'Please select a row first.')
                return

            seletedROW = indexes[0].row()
            X_test_CSV = self.CSV_df.iloc[[seletedROW]]
            self.df_Test_Case = X_test_CSV.drop(['target'], axis=1)

            model = pandasModel(self.df_Test_Case)
            self.patient_data_tableView.setModel(model)
        except Exception as e:
            self.qm.critical(self, 'Error', f"An error occurred: {str(e)}")

    @pyqtSlot()
    def BrowseButtonSlotFunction(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open CSV File', '.\\', "CSV Files (*.csv)")
        if fname:
            self.Display_CSV_FILE(fname)
        else:
            print("No Valid File selected.")

    def Display_CSV_FILE(self, fname):
        self.CSV_df = pd.read_csv(fname, sep=";")
        model = pandasModel(self.CSV_df)
        self.tableView.setModel(model)

    @pyqtSlot()
    def DetectionButtonSlotFunction(self):
        # print(self.data_array)

        self.selected_algo = str(self.Algo_Selector_ComboBox.currentText())

        if self.selected_algo == 'ML - KNN':
            self.KNN_Testing()

        elif self.selected_algo == 'ML - SVM':
            self.SVM_Testing()

        elif self.selected_algo == 'ML - Naive Bayes':
            self.NaiveBayes_Testing()

        elif self.selected_algo == 'ML - Decision Tree':
            self.DecisionTree_Testing()

        elif self.selected_algo == 'ML - Logistic Regression':
            self.LogisticRegression_Testing()

        elif self.selected_algo == 'EL - Random Forest':
            self.RandomForest_Testing()

        elif self.selected_algo == 'EL - Stochastic Gradient Boosting':
            self.SGB_Testing()

    def KNN_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/KNNClassifier.pkl', 'rb') as f:
            KNN_classifier = pickle.load(f)

        result = KNN_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def SVM_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/SVMclassifier.pkl', 'rb') as f:
            SVM_classifier = pickle.load(f)

        result = SVM_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def NaiveBayes_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/GNBclassifier.pkl', 'rb') as f:
            NB_classifier = pickle.load(f)

        result = NB_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def DecisionTree_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/DTCclassifier.pkl', 'rb') as f:
            DT_classifier = pickle.load(f)

        result = DT_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def LogisticRegression_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/LRclassifier.pkl', 'rb') as f:
            LR_classifier = pickle.load(f)

        result = LR_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def RandomForest_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/RFclassifier.pkl', 'rb') as f:
            RF_classifier = pickle.load(f)

        result = RF_classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def SGB_Testing(self):

        test_data = self.df_Test_Case.iloc[[0]].to_numpy()
        test_data_new = Normalizer().fit_transform(test_data)

        with open('./models/GBclassifier.pkl', 'rb') as f:
            GB_Classifier = pickle.load(f)

        result = GB_Classifier.predict(test_data_new)
        print(result)

        if result:
            self.label.setText('Heart Disease\nDetected')
            self.Show_Cardiac_Specialists_on_MAP()

        else:
            self.label.setText('Heart Disease\nNot Detected')

    def Show_Cardiac_Specialists_on_MAP(self):
        url = "https://www.google.co.in/maps/search/cardiac+specialists+near+me"
        webbrowser.open_new(url)

    @pyqtSlot()
    def ExitButtonSlotFunction(self):
        QApplication.instance().quit()


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None

    ''' ------------------------ MAIN Function ------------------------- '''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = CardioVascular_Disease_Prediction()
    window.show()
    sys.exit(app.exec_())
