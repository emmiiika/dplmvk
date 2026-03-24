import sys
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia
import cv2
from HandAnnotation import HandAnnotation
import numpy as np
import os
import random
from Scoring import Scoring


# ANSI escape codes for colored terminal output
HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"

FOLDER = "../videa/"  # Path to the folder containing reference videos
ANNOTATED_FOLDER = "../videa/.annotated/"  # Path to save annotated reference videos


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

        os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

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

        # Organize widgets into a grid layout
        self.layout = QtWidgets.QGridLayout()  # type: ignore
        self.layout.addWidget(self.webcam, 1, 0, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.score, 2, 1, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.gestureVideo, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore

        self.setLayout(self.layout)  # type: ignore

        print(f"{GREEN}✓{ENDC} UI setup complete.")

    def setupCamera(self):
        """Initialize webcam capture, configure hand annotation processor, and start refresh timer."""
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.videoSize.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.videoSize.height())

        # Initialize hand detection and annotation logic
        self.webcamAnnotation = HandAnnotation(self.capture)

        # Setup timer to refresh the video feed at ~33 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.displayVideoStream)
        self.timer.start(30)

        print(f"{GREEN}✓{ENDC} Camera setup complete.")

    def displayVideoStream(self):
        """Capture a frame from the webcam, annotate it with hand landmarks, and display in the UI.

        Called periodically by the camera timer (~33 FPS). Updates the webcam label with the annotated frame.
        """
        ret, frame = self.capture.read()

        if ret:
            image = self.webcamAnnotation.processSpecificFrame(frame, returnQt=True)
            if image is not None:
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
            Annotated videos are saved with '_annotated' suffix in the ANNOTATED_FOLDER.
        """

        # Generate output path by inserting '_annotated' before file extension
        filename = os.path.basename(referenceVideoPath)
        basename, ext = os.path.splitext(filename)
        outputPath = os.path.join(ANNOTATED_FOLDER, f"{basename}_annotated{ext}")

        if os.path.isfile(outputPath):
            print(f"Reference video {HEADER}already exists{ENDC}, loading from cache.")
            return cv2.VideoCapture(outputPath)

        return self.referenceAnnotation.createAnnotatedVideo(referenceVideoPath, outputPath)

    def loadReferenceVideo(self):
        """Load the reference gesture video from a path.

        Returns:
            Path to the reference video file."""

        if not os.path.exists(FOLDER) or not os.path.isdir(FOLDER):
            print(f"{FAIL}Error{ENDC}: folder does not exist or is not a directory: {FOLDER}")

            return None

        # Choose a random video from the folder, ignoring any already annotated videos
        referenceVideosList = os.listdir(FOLDER)

        idx = referenceVideosList.index(".annotated")
        referenceVideosList.pop(idx)

        randomVideo = random.choice(referenceVideosList)
        referenceVideoPath = os.path.join(FOLDER, randomVideo)

        # Process the reference video to create an annotated version (or load from cache if it already exists)
        nonAnnotatedVideo = cv2.VideoCapture(referenceVideoPath)
        self.referenceAnnotation = HandAnnotation(nonAnnotatedVideo)
        self.referenceVideo = self.annotateReferenceVideo(referenceVideoPath)

        self.scoring = Scoring(self.webcamAnnotation, self.referenceAnnotation)

        print(f"{GREEN}✓{ENDC} Reference video loaded from '{referenceVideoPath}'.")

        return referenceVideoPath

    def setupPlayer(self):
        """Initialize reference video playback and start the display refresh timer."""
        self.loadReferenceVideo()

        # Setup a separate timer for the reference video.
        self.refTimer = QtCore.QTimer()
        self.refTimer.timeout.connect(self.displayReferenceVideo)
        self.refTimer.start(30)  # Cca 30 FPS

        print(f"{GREEN}✓{ENDC} Player setup complete.")

    def displayReferenceVideo(self):
        """Display the next frame from the reference video, looping from the start when finished.

        Restarts from frame 0 when the video finishes. Converts from OpenCV BGR to Qt RGB format
        before displaying.
        """
        ret, frame = self.referenceVideo.read()

        if not ret:
            self.updateScore()  # Update the score display when the reference video finishes

            self.referenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.referenceVideo.read()

        if ret:
            # Convert BGR to RGB and display
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.referenceAnnotation.convertFrameToQtImage(rgbFrame)
            self.gestureVideo.setPixmap(QtGui.QPixmap.fromImage(image))

    def updateScore(self):
        """Update the score display label with the given score value."""

        currentScore = self.scoring.calculateScore()  # Calculate the score based on current annotation
        self.score.setText(f"Score: {currentScore:.2f}%")


if __name__ == "__main__":
    # Initialize and run the Qt application
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    print(f"{GREEN}✓{ENDC} Application initialized successfully.")

    sys.exit(app.exec())
