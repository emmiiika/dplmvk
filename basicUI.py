import sys
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia
import cv2
from HandAnnotation import HandAnnotation
import numpy as np


# Main application window class for gesture recognition and video playback
class Window(QtWidgets.QWidget):
    def __init__(self):
        """Initialize the main window, set dimensions, and trigger setup methods."""
        super().__init__()
        # Define video dimensions for webcam and reference video
        self.videoSize = QtCore.QSize(768, 576)
        self.gestureVideoSize = QtCore.QSize(640, 480)

        self.setupUi()
        self.setupCamera()
        self.setupPlayer()

    def setupUi(self):
        """Initialize and arrange UI widgets using a grid layout."""
        # Setup webcam display label
        self.webcam = QtWidgets.QLabel()
        self.webcam.setFixedSize(self.videoSize)

        # Setup score display label
        self.score = QtWidgets.QLabel()
        self.score.setText("Score: xx%")
        self.score.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # # Setup reference video player widget
        # self.gestureVideo = QtMultimediaWidgets.QVideoWidget()
        # self.gestureVideo.setFixedSize(self.gestureVideoSize)

        # Setup reference video player widget
        self.gestureVideo = QtWidgets.QLabel()
        self.gestureVideo.setFixedSize(self.gestureVideoSize)

        # Organize widgets into a grid
        self.layout = QtWidgets.QGridLayout()  # type: ignore
        self.layout.addWidget(self.webcam, 1, 0, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.score, 2, 1, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.gestureVideo, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore

        self.setLayout(self.layout)  # type: ignore

    def annotateFrame(self, frame):
        image = None
        if frame is not None:
            frame = self.annotation.processSpecificFrame(frame)

            # Convert OpenCV BGR format to RGB for Qt display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(  # type: ignore
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.strides[0],
                QtGui.QImage.Format.Format_RGB888,
            )
        return image

    def setupCamera(self):
        """Initialize OpenCV camera capture and set up the annotation processor."""
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.videoSize.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.videoSize.height())

        # Initialize hand detection and annotation logic
        self.annotation = HandAnnotation(self.capture)

        # Setup timer to refresh the video feed at ~33 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.displayVideoStream)
        self.timer.start(30)

    def displayVideoStream(self):
        """Process each camera frame, apply annotations, and update the UI label."""
        _, frame = self.capture.read()

        image = self.annotateFrame(frame)
        self.webcam.setPixmap(QtGui.QPixmap.fromImage(image))

    def loadReferenceVideo(self):
        referenceVideoPath = "../videa/dom - Snepeda (360p, h264).mp4"
        self.referenceVideo = cv2.VideoCapture(referenceVideoPath)

        return referenceVideoPath

    def annotateReferenceVideo(self):
        """Read, annotate and display one frame of the reference video."""
        ret, frame = self.referenceVideo.read()
        if not ret:  # TODO redo so it loads already annotated video, not reannotating
            # Loop video: restart from the beginning if it ends
            self.referenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.referenceVideo.read()

        if ret:
            image = self.annotateFrame(frame)
            if image:
                self.gestureVideo.setPixmap(QtGui.QPixmap.fromImage(image))

    def setupPlayer(self):
        """Initialize the reference video capture and start the animation timer."""
        self.loadReferenceVideo()

        # Setup a separate timer for the reference video.
        self.refTimer = QtCore.QTimer()
        self.refTimer.timeout.connect(self.displayReferenceVideo)
        self.refTimer.start(30)  # Cca 30 FPS

    def displayReferenceVideo(self):
        """Process each camera frame, apply annotations, and update the UI label."""
        _, frame = self.referenceVideo.read()

        image = self.annotateFrame(frame)
        self.gestureVideo.setPixmap(QtGui.QPixmap.fromImage(image))

    def saveVideo(self):
        """Trigger the video saving process in the annotation module."""
        self.annotation.saveVideo()


if __name__ == "__main__":
    # Initialize and run the Qt application
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    # Optional: Save video stream to file
    widget.saveVideo()

    sys.exit(app.exec())
