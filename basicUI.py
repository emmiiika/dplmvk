import sys
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia
import cv2
from HandAnnotation import HandAnnotation


# https://gist.github.com/bsdnoobz/8464000
class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # self.video_size = QtCore.QSize(320, 240)
        self.videoSize = QtCore.QSize(768, 576)
        self.gestureVideoSize = QtCore.QSize(640, 480)
        self.setupUi()
        self.setupCamera()
        self.setupPlayer()

    def setupUi(self):
        """Initialize widgets."""
        self.webcam = QtWidgets.QLabel()
        self.webcam.setFixedSize(self.videoSize)

        self.score = QtWidgets.QLabel()
        self.score.setText("Score: xx%")
        self.score.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.gestureVideo = QtMultimediaWidgets.QVideoWidget()
        self.gestureVideo.setFixedSize(self.gestureVideoSize)

        self.layout = QtWidgets.QGridLayout()  # type: ignore
        # self.layout.setContentsMargins(50, 50, 50, 50)
        # self.layout.setSpacing(25)

        self.layout.addWidget(self.webcam, 1, 0, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.score, 2, 1, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.layout.addWidget(self.gestureVideo, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore

        self.setLayout(self.layout)  # type: ignore

    def setupCamera(self):
        """Initialize camera."""
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.videoSize.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.videoSize.height())

        self.annotation = HandAnnotation(self.capture)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget."""
        _, frame = self.capture.read()
        frame = self.annotation.processFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(  # type: ignore
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        self.webcam.setPixmap(QtGui.QPixmap.fromImage(image))

    def setupPlayer(self):
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setSource(QtCore.QUrl.fromLocalFile("../videa/dom - Snepeda (360p, h264).mp4"))

        self.player.setVideoOutput(self.gestureVideo)
        self.player.setLoops(-1)
        self.player.play()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())
