import sys
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia
import cv2
from HandAnnotation import HandAnnotation
import numpy as np
import os

# ANSI escape codes for colored terminal output
HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"


# Main application window class for gesture recognition and video playback
class Window(QtWidgets.QWidget):
    """Main application window for hand gesture recognition and reference video playback.

    This class manages the GUI with webcam feed, score display, and reference video playback.
    Uses PySide6 for UI components and OpenCV for frame capture and annotation.
    """

    def __init__(self):
        """Initialize the main application window with video dimensions and UI components, trigger setup methods."""
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
        """Process a single frame with hand detection and convert to Qt-compatible image format.

        Args:
            frame: Input BGR frame from OpenCV.

        Returns:
            QImage object in RGB format ready for display, or None if frame is invalid.
        """
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
        """Initialize webcam capture, configure hand annotation processor, and start refresh timer."""
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
        """Capture a frame from the webcam, annotate it with hand landmarks, and display in the UI.

        Called periodically by the camera timer. Updates the webcam label with the annotated frame.
        """
        _, frame = self.capture.read()

        image = self.annotateFrame(frame)
        self.webcam.setPixmap(QtGui.QPixmap.fromImage(image))

    def annotateReferenceVideo(self, referenceVideoPath):
        """Process and cache an annotated version of the reference video.

        Checks if an annotated version already exists; if so, returns it. Otherwise,
        creates a new annotated video by processing the original frame-by-frame.

        Args:
            referenceVideoPath: Path to the input reference video file.

        Returns:
            cv2.VideoCapture object pointing to the annotated reference video (existing or newly created).

        Note:
            Annotated videos are saved with '_annotated' suffix in the same directory.
        """
        inputPath = referenceVideoPath.split(".")
        outputPath = ".." + inputPath[-2] + "_annotated." + inputPath[-1]

        if os.path.isfile(outputPath):
            print(f"Reference video {HEADER}already exists{ENDC}, loading the existing one.")

            return cv2.VideoCapture(outputPath)

        return self.annotation.createAnnotatedVideo(referenceVideoPath, outputPath)

    def loadReferenceVideo(self):
        """Load the reference gesture video from a path.

        Returns:
            Path to the reference video file."""
        referenceVideoPath = "../videa/dom - Snepeda (360p, h264).mp4"
        self.referenceVideo = self.annotateReferenceVideo(referenceVideoPath)

        print(f"{GREEN}Reference video loaded from '{referenceVideoPath}'{ENDC}.")

        return referenceVideoPath

    def setupPlayer(self):
        """Initialize reference video playback and start the display refresh timer."""
        self.loadReferenceVideo()

        # Setup a separate timer for the reference video.
        self.refTimer = QtCore.QTimer()
        self.refTimer.timeout.connect(self.displayReferenceVideo)
        self.refTimer.start(30)  # Cca 30 FPS

    def displayReferenceVideo(self):
        """Display the next frame from the reference video, looping when the end is reached.

        Restarts from frame 0 when the video finishes. Converts from OpenCV BGR to Qt RGB format
        before displaying.
        """
        _, frame = self.referenceVideo.read()

        if not _:
            self.referenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.referenceVideo.read()

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(  # type: ignore
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.strides[0],
                QtGui.QImage.Format.Format_RGB888,
            )
            self.gestureVideo.setPixmap(QtGui.QPixmap.fromImage(image))


if __name__ == "__main__":
    # Initialize and run the Qt application
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())
