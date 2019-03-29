import sys
import numpy as np
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5 import QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from tomograph import ComputerTomography
from dicom.dataset import Dataset, FileDataset
import dicom
import time
import datetime


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        main_layout = QVBoxLayout()
        layout = QGridLayout()
        self.img_figure = plt.figure(None, figsize=(16, 16))
        self.img_canvas = FigureCanvas(self.img_figure)

        self.radon_figure = plt.figure(None, figsize=(16, 16))
        self.radon_canvas = FigureCanvas(self.radon_figure)

        self.reverse_figure = plt.figure(None, figsize=(16, 16))
        self.reverse_canvas = FigureCanvas(self.reverse_figure)

        self.figures = [self.img_figure, self.radon_figure, self.reverse_figure]
        self.canvases = [self.img_canvas, self.radon_canvas, self.reverse_canvas]
        self.titles = ["Image", "Sinogram", "Reverse"]
        self.images = [np.array([[0]]), np.array([[0]]), np.array([[0]])]

        self.load_button = QPushButton("Load imgage/dicom")
        self.load_button.clicked.connect(self._load_img)
        self.sinogram_button = QPushButton("Radon transform")
        self.sinogram_button.clicked.connect(self._radon_transform)
        self.sinogram_button.setEnabled(False)
        self.reverse_button = QPushButton("Reverse Radon transform")
        self.reverse_button.clicked.connect(self._iradon_transform)
        self.reverse_button.setEnabled(False)

        self.mse_label = QtWidgets.QLabel("MSE: ")
        layout.addWidget(self.mse_label, 0, 3)
        button_grid = QGridLayout()
        layout.addLayout(button_grid, 0, 1)
        button_grid.addWidget(self.load_button, 0, 1)
        button_grid.addWidget(self.sinogram_button, 0, 2)
        button_grid.addWidget(self.reverse_button, 0, 3)
        layout.addWidget(self.img_canvas, 1, 1)

        layout.addWidget(self.radon_canvas, 1, 2)

        layout.addWidget(self.reverse_canvas, 1, 3)
        main_layout.addLayout(layout)

        onlyInt = QtGui.QIntValidator(0, 10000)
        self.number_emitters_edit = QtWidgets.QLineEdit()
        self.number_emitters_edit.setText("180")
        self.number_emitters_edit.setValidator(onlyInt)
        number_emitters_label = QtWidgets.QLabel("Detector number: ")

        self.number_steps_edit = QtWidgets.QLineEdit()
        self.number_steps_edit.setText("180")
        self.number_steps_edit.setValidator(onlyInt)
        number_steps_label = QtWidgets.QLabel("Step number: ")

        onlyDouble = QtGui.QDoubleValidator(0.0, 360.0, 0)
        self.emitter_width = QtWidgets.QLineEdit()
        self.emitter_width.setText("180")
        self.emitter_width.setValidator(onlyDouble)
        emitters_width_label = QtWidgets.QLabel("Detector width (0-360> :")

        backprojection_label = QtWidgets.QLabel("Backprojection filter: ")
        self.backprojection_checkbox = QtWidgets.QCheckBox()
        iterative_label = QtWidgets.QLabel("Iterative:")
        self.iterative_checkbox = QtWidgets.QCheckBox()
        options = QGridLayout()
        options.addWidget(number_emitters_label, 0, 0)
        options.addWidget(self.number_emitters_edit, 0, 1)
        options.addWidget(number_steps_label, 0, 2)
        options.addWidget(self.number_steps_edit, 0, 3)
        options.addWidget(emitters_width_label, 0, 4)
        options.addWidget(self.emitter_width, 0, 5)
        options.addWidget(backprojection_label, 1, 0)
        options.addWidget(self.backprojection_checkbox, 1, 1)
        options.addWidget(iterative_label, 1, 2)
        options.addWidget(self.iterative_checkbox, 1, 3)

        self.patient_name_edit = QtWidgets.QLineEdit()
        patient_name_label = QtWidgets.QLabel("Patient name: ")

        self.comment_edit = QtWidgets.QLineEdit()
        comment_label = QtWidgets.QLabel("Comment: ")

        options.addWidget(patient_name_label, 2, 0)
        options.addWidget(self.patient_name_edit, 2, 1)
        options.addWidget(comment_label, 2, 2)
        options.addWidget(self.comment_edit, 2, 3)
        self.dicom_button = QPushButton("Save as dicom")
        self.dicom_button.clicked.connect(self._save_dicom)
        self.dicom_button.setEnabled(False)
        options.addWidget(self.dicom_button, 2, 4)

        main_layout.addLayout(options)
        options.setContentsMargins(20, 20, 20, 20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self._replot_all()
        self.setLayout(main_layout)

    def _save_dicom(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        fname = QtWidgets.QFileDialog.getSaveFileName(self, "", options=options)[0]
        if fname != "":
            image = self.images[2].copy()
            # Resize to power of two
            res = []
            col_power = 2
            while col_power < image.shape[1]:
                col_power *= 2
            row_power = 2
            while row_power < image.shape[0]:
                row_power *= 2

            for row in image:
                res.append(np.concatenate((row, np.array([0] * (col_power - row.shape[0])))))
            for i in range(image.shape[0], row_power):
                res.append([0]*(col_power))
            image = np.array(res)
            name = self.patient_name_edit.text()
            comment = self.comment_edit.text()
            print(image.shape)
            if fname.split(".")[-1] != "dcm":
                fname += ".dcm"
            write_dicom(image * 255, fname, name, comment)

    def _iradon_transform(self):
        self.sinogram_button.setEnabled(False)
        self.reverse_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.dicom_button.setEnabled(False)

        app.processEvents()
        iterative = self.iterative_checkbox.checkState()
        if iterative:
            fun = lambda x: self._update_replot_all(2, x)
            self.ct.iradon_transform_iterative(self.images[1], fun)
        else:
            self.images[2] = self.ct.iradon_transform(self.images[1])
            self._replot_all()

        self.mse_label.setText("MSE: {0}".format(np.mean(self.images[0] / 255 - self.images[2]) ** 2))
        self.sinogram_button.setEnabled(True)
        self.reverse_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.dicom_button.setEnabled(True)

    def _update_replot_all(self, i, image):
        app.processEvents()
        self.images[i] = image
        self._replot_all()

    def _radon_transform(self):
        self.sinogram_button.setEnabled(False)
        self.reverse_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.dicom_button.setEnabled(False)
        app.processEvents()
        number_step = int(self.number_steps_edit.text())
        number_emitters = int(self.number_emitters_edit.text())
        scan_width = float(self.emitter_width.text())
        iterative = self.iterative_checkbox.checkState()
        backprojection = self.backprojection_checkbox.checkState()
        self.ct = ComputerTomography(number_step, number_emitters, scan_width / 360 * 2 * np.pi)
        if iterative:
            fun = lambda x: self._update_replot_all(1, x)
            self.ct.radon_transform_iterative(self.images[0], backprojection, fun)
        else:
            self.images[1] = self.ct.radon_transform(self.images[0], backprojection)
            self._replot_all()

        self.sinogram_button.setEnabled(True)
        self.reverse_button.setEnabled(True)
        self.load_button.setEnabled(True)

    def _load_img(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        fname = QtWidgets.QFileDialog.getOpenFileName(self, "", options=options)[0]
        if fname != "":

            if fname.split(".")[-1] == "dcm":

                dc = dicom.read_file(fname)
                self.images[0] = np.fromstring(dc.PixelData, dtype=np.int16).reshape((dc.Rows, dc.Columns))
                self.comment_edit.setText(str(dc.ImageComments))
                self.patient_name_edit.setText(str(dc.PatientName))

            else:
                self.images[0] = imread(fname, flatten=True)
                self.comment_edit.setText("")
                self.patient_name_edit.setText("")

            self.dicom_button.setEnabled(False)
            self.sinogram_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.images[1] = np.array([[0]])
            self.images[2] = np.array([[0]])

            self._replot_all()

    def _replot_all(self):
        for figure, canvas, title, img in zip(self.figures, self.canvases, self.titles, self.images):
            figure.clear()

            ax = figure.add_subplot(111)
            ax.set_title(title)
            ax.matshow(img, cmap="gray")
            canvas.draw()

    def plot(self, img, figure, canvas):
        figure.clear()
        ax = figure.add_subplot(111)
        ax.matshow(img, cmap="gray")
        canvas.draw()


def write_dicom(pixel_array, filename, patient_name, comment):
    """ From stackoverflow
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b'\x00' * 128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-', '')
    ds.ContentTime = str(time.time())  # milliseconds since the epoch
    ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
    ds.PatientName = patient_name
    ds.ImageComments = comment
    ## These are the necessary imaging components of the FileDataset object.

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = b'\\x00\\x00'
    ds.LargestImagePixelValue = b'\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return


image = np.zeros((16,16))
# Resize to power of two
res = []
col_power = 2
while col_power < image.shape[1]:
    col_power *= 2
row_power = 2
while row_power < image.shape[0]:
    row_power *= 2
print(row_power, col_power)
for row in image:
    res.append(np.concatenate((row, np.array([0] * (col_power - row.shape[0])))))
for i in range(image.shape[0], row_power):
    res.append([0] * (col_power))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
