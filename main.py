import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Importing plotting functions
from beamProfile import plot_beam_profiles  # General image profile plotting
from beamProfile_npy import plot_beam_profiles_npy  # npy image profile plotting

class BeamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Profile Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # GUI Components
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(460, 280)

        # Buttons Layout
        self.button_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture Image", self)
        self.open_button = QPushButton("Open Image", self)
        self.profile_button = QPushButton("Plot Beam Profile", self)

        self.button_layout.addWidget(self.capture_button)
        self.button_layout.addWidget(self.open_button)
        self.button_layout.addWidget(self.profile_button)

        # Parameters Layout
        self.params_layout_hor = QHBoxLayout()
        self._add_param_label("Amp_hor:", self.params_layout_hor)
        self._add_param_label("Mean_hor:", self.params_layout_hor)
        self._add_param_label("Sigma_hor:", self.params_layout_hor)
        self._add_param_label("FWHM_hor:", self.params_layout_hor)

        # Parameters for vertical profiles
        self.params_layout_vert = QHBoxLayout()
        self._add_param_label("Amp_vert:", self.params_layout_vert)
        self._add_param_label("Mean_vert:", self.params_layout_vert)
        self._add_param_label("Sigma_vert:", self.params_layout_vert)
        self._add_param_label("FWHM_vert:", self.params_layout_vert)

        # Horizontal and Vertical Profile Canvases
        self.figure_hor, self.ax_hor = plt.subplots()
        self.canvas_hor = FigureCanvas(self.figure_hor)
        self.canvas_hor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas_hor.setFixedHeight(280)

        self.figure_vert, self.ax_vert = plt.subplots()
        self.canvas_vert = FigureCanvas(self.figure_vert)
        self.canvas_vert.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.canvas_vert.setFixedWidth(460)

        # Image Layout with Profiles
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_label)
        self.image_layout.addWidget(self.canvas_vert)

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.params_layout_hor)
        self.main_layout.addLayout(self.params_layout_vert)
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addWidget(self.canvas_hor)

        # Set the main layout to the central widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Connect buttons to actions
        self.capture_button.clicked.connect(self.capture_image)
        self.open_button.clicked.connect(self.load_image)
        self.profile_button.clicked.connect(self.plot_beam_profile)

        self.captured_image = None
        self.loaded_image = None
        self.file_path = None  # Store file path for profile plotting

    def _add_param_label(self, text, layout):
        label = QLabel(text, self)
        layout.addWidget(label)
        value_label = QLabel(self)
        layout.addWidget(value_label)
        setattr(self, f"{text.lower().replace(':', '').strip()}_value", value_label)

    def capture_image(self):
        """Capture an image from the webcam."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.display_image(self.captured_image)

    def display_image(self, image):
        qformat = QImage.Format_Indexed8
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def load_image(self):
        """Open an image from disk and decide processing method based on file type."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Beam Image", "", "Image Files (*.png *.jpg *.bmp *.npy *.pgm)")
        if path:
            self.file_path = path  # Store the path for later access in plot_beam_profile
            if path.endswith('.npy'):
                # Load and process .npy image
                image = np.load(path)
            else:
                # Load general image types
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                self.loaded_image = image
                self.display_image(image)

    def plot_beam_profile(self):
        """Plot the beam profiles and update the GUI with Gaussian parameters."""
        if self.file_path and self.file_path.endswith('.npy'):
            # Use the .npy specific beam profile code
            params_hor, params_vert = plot_beam_profiles_npy(self.loaded_image, self.ax_hor, self.ax_vert)
        else:
            # Use the general beam profile code
            params_hor, params_vert = plot_beam_profiles(self.loaded_image, self.ax_hor, self.ax_vert)

        # Extract and display parameters
        _, mean_hor, sigma_hor, _ = params_hor
        fwhm_hor = 2.355 * sigma_hor
        _, mean_vert, sigma_vert, _ = params_vert
        fwhm_vert = 2.355 * sigma_vert

        # Update GUI with calculated parameters
        self.mean_hor_value.setText(f"{mean_hor:.2f}")
        self.sigma_hor_value.setText(f"{sigma_hor:.2f}")
        self.fwhm_hor_value.setText(f"{fwhm_hor:.2f}")
        self.mean_vert_value.setText(f"{mean_vert:.2f}")
        self.sigma_vert_value.setText(f"{sigma_vert:.2f}")
        self.fwhm_vert_value.setText(f"{fwhm_vert:.2f}")

        # Redraw the canvases to display the plots
        self.ax_vert.invert_yaxis()  # Ensure y-axis is inverted for vertical plot
        self.canvas_hor.draw()
        self.canvas_vert.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamGUI()
    window.show()
    sys.exit(app.exec_())
