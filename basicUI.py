import sys
from PySide6 import QtCore, QtWidgets, QtGui
import cv2
from HandAnnotation import HandAnnotation
import numpy as np
import os
import random
from Scoring import Scoring
import time
import json
import copy


# ANSI escape codes for colored terminal output
HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"

# Base playback speed (FPS) for the reference video timer
BASE_FPS = 30
# Speed steps (multipliers) available for playback
SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


class TrimProgressBar(QtWidgets.QWidget):
    """A progress bar that also draws trim-start and trim-end markers."""

    BAR_COLOR = QtGui.QColor("#6666cc")
    TRACK_COLOR = QtGui.QColor("#2a2a3e")
    MARKER_COLOR = QtGui.QColor("#ffaa00")

    MARKER_HIT_PX = 6  # pixel tolerance for hover detection

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0  # 0-1000
        self._trimStart = 0  # 0-1000 position of start marker
        self._trimEnd = 1000  # 0-1000 position of end marker
        self._markersVisible = False
        self.setFixedHeight(10)
        self.setMouseTracking(True)

    def setValue(self, v: int):
        self._value = max(0, min(1000, v))
        self.update()

    def reset(self):
        """Clear the bar and hide markers (e.g. while waiting for annotation)."""
        self._value = 0
        self._markersVisible = False
        self.update()

    def setTrimMarkers(self, start: int, end: int):
        """Set trim markers in the same 0-1000 scale as value."""
        self._trimStart = max(0, min(1000, start))
        self._trimEnd = max(0, min(1000, end))
        self._markersVisible = True
        self.update()

    def mouseMoveEvent(self, event):
        x = event.position().x()
        w = self.width()
        startX = self._trimStart / 1000 * w
        endX = self._trimEnd / 1000 * w
        if self._markersVisible and abs(x - startX) <= self.MARKER_HIT_PX:
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), "Start of movement", self)
        elif self._markersVisible and abs(x - endX) <= self.MARKER_HIT_PX:
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), "End of movement", self)
        else:
            QtWidgets.QToolTip.hideText()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = h // 2

        # Track
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(self.TRACK_COLOR)
        p.drawRoundedRect(0, 0, w, h, r, r)

        # Filled chunk (0 → current value)
        fillRight = int(self._value / 1000 * w)
        if fillRight > 0:
            p.setBrush(self.BAR_COLOR)
            p.drawRoundedRect(0, 0, fillRight, h, r, r)

        # Trim markers — vertical lines
        if self._markersVisible:
            p.setPen(QtGui.QPen(self.MARKER_COLOR, 2))
            for pos in (self._trimStart, self._trimEnd):
                x = int(pos / 1000 * w)
                p.drawLine(x, 0, x, h)

        p.end()


class AnnotationWorker(QtCore.QThread):
    """Background thread that runs createAnnotatedVideo so the UI stays responsive."""

    finished = QtCore.Signal()

    def __init__(self, referenceAnnotation, videoPath, outputPath, parent=None):
        super().__init__(parent)
        self._referenceAnnotation = referenceAnnotation
        self._videoPath = videoPath
        self._outputPath = outputPath

    def run(self):
        self._referenceAnnotation.createAnnotatedVideo(self._videoPath, self._outputPath)
        self.finished.emit()


# Main application window class for gesture recognition and video playback
class Window(QtWidgets.QWidget):
    """Main application window for hand gesture recognition and reference video playback.

    This class manages the GUI with webcam feed, score display, and reference video playback.
    Uses PySide6 for UI components and OpenCV for frame capture and annotation.
    """

    FOLDER = "../videa/"  # Path to the folder containing reference videos
    ANNOTATED_FOLDER = "../videa/.annotated/"  # Path to save annotated reference videos
    SAMPLING_RATE = 0.04  # seconds (25 FPS) for collecting user landmarks during tracking
    RECORDING_PATH = "../videa/.recorded/user_recording.avi"

    def __init__(self):
        """Initialize the main application window with video dimensions and UI components, trigger setup methods."""
        super().__init__()
        # Define video dimensions for webcam and reference video
        self.videoSize = QtCore.QSize(640, 480)
        self.gestureVideoSize = QtCore.QSize(640, 480)

        # Tracking state for user movement comparison
        self.isTracking = False
        self.startTime = 0.0
        self.nextSampleTime = 0.0
        self.userLandmarksTimestamped = []

        # Pause between reference video loops (seconds)
        self.referenceVideoLoopPause = 3.0
        self.referenceVideoPauseUntil = 0.0

        # Reference video queue and navigation
        self.videoQueue = []
        self.videoQueueIndex = 0

        # Playback state
        self.refPaused = False

        # Recording state
        self.isRecording = False
        self.videoWriter = None
        self.lastRecordingPath = None

        # Annotated reference video capture (pre-rendered with landmarks drawn)
        self.annotatedReferenceVideo = None

        # Background annotation worker
        self._annotationWorker = None

        # Playback mode: when True, the left panel shows the recorded video instead of webcam
        self.isPlaybackMode = False
        self.playbackCapture = None

        # Annotation visibility
        self.showAnnotations = True

        # Speed control
        self.speedIndex = SPEED_STEPS.index(1.0)

        self.setupUi()
        self.setupCamera()
        self.setupPlayer()

        os.makedirs(self.ANNOTATED_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.RECORDING_PATH)), exist_ok=True)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def setupUi(self):
        """Initialize and arrange UI widgets using a vertical layout with a control panel."""

        # --- Video labels ---
        self.webcam = QtWidgets.QLabel()
        self.webcam.setFixedSize(self.videoSize)
        self.webcam.setStyleSheet("background-color: #1a1a2e;")

        self.gestureVideo = QtWidgets.QLabel()
        self.gestureVideo.setFixedSize(self.gestureVideoSize)
        self.gestureVideo.setStyleSheet("background-color: #1a1a2e;")

        # --- Loading overlay (shown while annotation is being processed) ---
        self._loadingPage = QtWidgets.QWidget()
        self._loadingPage.setFixedSize(self.gestureVideoSize)
        self._loadingPage.setStyleSheet("background-color: #0d0d1a;")
        _ll = QtWidgets.QVBoxLayout(self._loadingPage)
        _ll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        _loadingLabel = QtWidgets.QLabel("Annotating video…")
        _loadingLabel.setStyleSheet("color: #c0c0e0; font-size: 18px; font-weight: bold; background: transparent;")
        _loadingLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._annotationSpinner = QtWidgets.QProgressBar()
        self._annotationSpinner.setRange(0, 0)  # indeterminate animation
        self._annotationSpinner.setFixedWidth(300)
        self._annotationSpinner.setFixedHeight(8)
        self._annotationSpinner.setTextVisible(False)
        self._annotationSpinner.setStyleSheet(
            "QProgressBar { background-color: #2a2a3e; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #6666cc; border-radius: 4px; }"
        )
        _ll.addWidget(_loadingLabel)
        _ll.addSpacing(14)
        _ll.addWidget(self._annotationSpinner, 0, QtCore.Qt.AlignmentFlag.AlignCenter)

        # Stack: index 0 = normal video, index 1 = loading screen
        self.refStack = QtWidgets.QStackedWidget()
        self.refStack.setFixedSize(self.gestureVideoSize)
        self.refStack.addWidget(self.gestureVideo)  # index 0
        self.refStack.addWidget(self._loadingPage)  # index 1

        # --- Score label (centred, between the two videos) ---
        self.score = QtWidgets.QLabel("Score: --%")
        self.score.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.score.setMinimumWidth(160)
        self.score.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #888888;"
            "background-color: #2a2a3e; border-radius: 8px; padding: 8px 16px;"
        )

        # --- Progress bar under the reference video ---
        self.refProgressBar = TrimProgressBar()
        self.refProgressBar.setFixedWidth(self.gestureVideoSize.width())

        # Wrap reference video stack + progress bar in a column
        refCol = QtWidgets.QVBoxLayout()
        refCol.setSpacing(4)
        refCol.addWidget(self.refStack)
        refCol.addWidget(self.refProgressBar)

        # --- Top row: [webcam] [score] [reference+bar] ---
        topRow = QtWidgets.QHBoxLayout()
        topRow.setSpacing(16)
        topRow.addWidget(self.webcam)

        scoreCol = QtWidgets.QVBoxLayout()
        scoreCol.addStretch()
        scoreCol.addWidget(self.score)
        scoreCol.addStretch()
        topRow.addLayout(scoreCol)

        topRow.addLayout(refCol)

        # --- Control panel ---
        controlPanel = self._buildControlPanel()

        # --- Root layout ---
        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addLayout(topRow)
        root.addWidget(controlPanel)
        self.setLayout(root)

        self.setStyleSheet("background-color: #12121f; color: #e0e0e0;")
        self.setWindowTitle("Gesture Trainer")

        print(f"{GREEN}✓{ENDC} UI setup complete.")

    def _buildControlPanel(self):
        """Build and return the control panel widget with all playback/record buttons."""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("background-color: #1e1e30; border-radius: 10px; padding: 4px;")

        btnStyle = (
            "QPushButton {"
            "  background-color: #2e2e50; color: #e0e0e0;"
            "  border: 1px solid #444466; border-radius: 6px;"
            "  padding: 6px 14px; font-size: 18px;"
            "}"
            "QPushButton:hover { background-color: #3e3e70; }"
            "QPushButton:pressed { background-color: #555590; }"
            "QPushButton:checked { background-color: #4a4a90; border: 1px solid #8888cc; }"
            "QPushButton:disabled { background-color: #1e1e30; color: #555566; border: 1px solid #333344; }"
        )

        def btn(text, tooltip):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tooltip)
            b.setStyleSheet(btnStyle)
            b.setFixedHeight(42)
            return b

        self.btnPrev = btn("⏮", "Previous reference video")
        self.btnPlayPause = btn("⏸", "Pause / Play reference video")
        self.btnNext = btn("⏭", "Next reference video")
        self.btnRecord = btn("⏺", "Start / Stop recording")
        self.btnPlayback = btn("📽", "Play last recorded video")
        self.btnAnnotations = btn("👁", "Toggle annotation visibility")
        self.btnAnnotations.setCheckable(True)
        self.btnAnnotations.setChecked(True)

        # Speed control: [−] [1.0×] [+]
        self.btnSpeedDown = btn("−", "Decrease playback speed")
        self.btnSpeedDown.setFixedWidth(36)
        self.speedLabel = QtWidgets.QLabel(f"{SPEED_STEPS[self.speedIndex]:.2g}×")
        self.speedLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.speedLabel.setStyleSheet("color: #e0e0e0; font-size: 16px; min-width: 44px;")
        self.btnSpeedUp = btn("+", "Increase playback speed")
        self.btnSpeedUp.setFixedWidth(36)

        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        layout.addWidget(self.btnPrev)
        layout.addWidget(self.btnPlayPause)
        layout.addWidget(self.btnNext)
        layout.addSpacing(12)
        layout.addWidget(self.btnRecord)
        layout.addWidget(self.btnPlayback)
        layout.addSpacing(12)
        layout.addWidget(self.btnAnnotations)
        layout.addStretch()
        layout.addWidget(self.btnSpeedDown)
        layout.addWidget(self.speedLabel)
        layout.addWidget(self.btnSpeedUp)

        # Connect signals
        self.btnPrev.clicked.connect(self.onPrevVideo)
        self.btnPlayPause.clicked.connect(self.onPlayPause)
        self.btnNext.clicked.connect(self.onNextVideo)
        self.btnRecord.clicked.connect(self.onRecord)
        self.btnPlayback.clicked.connect(self.onPlayback)
        self.btnAnnotations.toggled.connect(self.onToggleAnnotations)
        self.btnSpeedDown.clicked.connect(self.onSpeedDown)
        self.btnSpeedUp.clicked.connect(self.onSpeedUp)

        return panel

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

    # ------------------------------------------------------------------
    # Control panel callbacks
    # ------------------------------------------------------------------

    def onPrevVideo(self):
        """Navigate to the previous video in the queue."""
        if not self.videoQueue:
            return
        self.videoQueueIndex = (self.videoQueueIndex - 1) % len(self.videoQueue)
        self._loadVideoAtIndex(self.videoQueueIndex)

    def onPlayPause(self):
        """Toggle pause/play of the reference video."""
        self.refPaused = not self.refPaused
        self.btnPlayPause.setText("▶" if self.refPaused else "⏸")
        print(f"{BLUE}Toggled play/pause.{ENDC}")

    def onNextVideo(self):
        """Advance to the next video in the queue."""
        if not self.videoQueue:
            return
        self.videoQueueIndex = (self.videoQueueIndex + 1) % len(self.videoQueue)
        self._loadVideoAtIndex(self.videoQueueIndex)

    def onRecord(self):
        """Start or stop recording the webcam stream."""
        if self.isRecording:
            self._stopRecording()
        else:
            self._startRecording()

    def _startRecording(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.RECORDING_PATH)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore
        fps = 30.0
        size = (self.videoSize.width(), self.videoSize.height())
        self.videoWriter = cv2.VideoWriter(self.RECORDING_PATH, fourcc, fps, size)
        self.isRecording = True
        self.lastRecordingPath = self.RECORDING_PATH
        self.btnRecord.setText("⏹")
        self.btnRecord.setStyleSheet(self.btnRecord.styleSheet().replace("#2e2e50", "#6e2020"))
        print(f"{BLUE}Recording started.{ENDC}")

    def _stopRecording(self):
        if self.videoWriter is not None:
            self.videoWriter.release()
            self.videoWriter = None
        self.isRecording = False
        self.btnRecord.setText("⏺")
        # Reset button style
        self.btnRecord.setStyleSheet(self.btnRecord.styleSheet().replace("#6e2020", "#2e2e50"))
        print(f"{BLUE}Recording stopped, saved to '{self.RECORDING_PATH}'.{ENDC}")

    def onPlayback(self):
        """Toggle playback of the last recorded video in the webcam slot."""
        if self.isPlaybackMode:
            # Stop playback, go back to live webcam
            self.isPlaybackMode = False
            if self.playbackCapture is not None:
                self.playbackCapture.release()
                self.playbackCapture = None
            self.btnPlayback.setText("📽")
            print(f"{BLUE}Playback stopped, resuming live webcam.{ENDC}")
        else:
            if self.lastRecordingPath is None or not os.path.isfile(self.lastRecordingPath):
                print(f"{WARNING}Warning:{ENDC} No recorded video to play back.")
                return
            self.playbackCapture = cv2.VideoCapture(self.lastRecordingPath)
            self.isPlaybackMode = True
            self.btnPlayback.setText("📷")
            print(f"{BLUE}Playing back recorded video.{ENDC}")

    def onToggleAnnotations(self, checked: bool):
        """Show or hide hand annotations on both video streams."""
        self.showAnnotations = checked
        self.btnAnnotations.setText("👁" if checked else "🚫")
        print(f"{BLUE}Toggled annotations.{ENDC}")

    def onSpeedDown(self):
        """Decrease reference video playback speed."""
        if self.speedIndex > 0:
            self.speedIndex -= 1
            self._applySpeed()

    def onSpeedUp(self):
        """Increase reference video playback speed."""
        if self.speedIndex < len(SPEED_STEPS) - 1:
            self.speedIndex += 1
            self._applySpeed()

    def _applySpeed(self):
        speed = SPEED_STEPS[self.speedIndex]
        self.speedLabel.setText(f"{speed:.2g}×")
        interval = int(1000 / (BASE_FPS * speed))
        self.refTimer.setInterval(interval)
        print(f"{BLUE}Reference video speed set to {speed:.2g}× ({interval} ms/frame).{ENDC}")

    def displayVideoStream(self):
        """Capture a frame from the webcam (or playback file), annotate, and display in the UI.

        Called periodically by the camera timer (~33 FPS). When in playback mode the left
        panel shows frames from the last recorded video instead of the live webcam.
        Collects user landmarks for scoring when tracking is active.
        """
        if self.isPlaybackMode and self.playbackCapture is not None:
            ret, frame = self.playbackCapture.read()
            if not ret:
                # Loop playback from start
                self.playbackCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.playbackCapture.read()
            if ret:
                if self.showAnnotations:
                    image = self.webcamAnnotation.processSpecificFrame(frame, returnQt=True)
                else:
                    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = self.webcamAnnotation.convertFrameToQtImage(rgbFrame)
                if image is not None:
                    self.webcam.setPixmap(QtGui.QPixmap.fromImage(image))
            return

        # --- Live webcam mode ---
        ret, frame = self.capture.read()

        if ret:
            if self.showAnnotations:
                image = self.webcamAnnotation.processSpecificFrame(frame, returnQt=True)
            else:
                rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self.webcamAnnotation.convertFrameToQtImage(rgbFrame)

            if image is not None:
                self.webcam.setPixmap(QtGui.QPixmap.fromImage(image))

            # Write frame to video file when recording
            if self.isRecording and self.videoWriter is not None:
                self.videoWriter.write(frame)

            # Collect user landmarks at the configured sampling rate when tracking is active
            if self.isTracking:
                timestamp = time.time() - self.startTime
                if timestamp >= self.nextSampleTime:
                    self.userLandmarksTimestamped.append(
                        (timestamp, copy.deepcopy(self.webcamAnnotation.handLandmarksList))
                    )
                    self.nextSampleTime += self.SAMPLING_RATE

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
        outputPath = os.path.join(self.ANNOTATED_FOLDER, f"{basename}_annotated{ext}")
        landmarksPath = os.path.splitext(outputPath)[0] + "_handLandmarks.json"

        if os.path.isfile(outputPath):
            print(f"Reference video {HEADER}already exists{ENDC}, loading from cache.")

            # Ensure reference landmarks are loaded when using cached annotated video.
            loaded = self.referenceAnnotation.loadLandmarksFromFile(landmarksPath)
            if loaded is None:
                print(
                    f"{WARNING}Warning{ENDC}: Cached landmarks missing for '{outputPath}'. "
                    "Rebuilding annotated video and landmarks."
                )
                return self.referenceAnnotation.createAnnotatedVideo(referenceVideoPath, outputPath)

            return cv2.VideoCapture(outputPath)

        return self.referenceAnnotation.createAnnotatedVideo(referenceVideoPath, outputPath)  # type: ignore

    def _buildVideoQueue(self):
        """Scan FOLDER and build a shuffled list of available reference video paths."""
        if not os.path.exists(self.FOLDER) or not os.path.isdir(self.FOLDER):
            print(f"{FAIL}Error{ENDC}: folder does not exist or is not a directory: {self.FOLDER}")
            return []

        allFiles = os.listdir(self.FOLDER)
        videos = [
            os.path.join(self.FOLDER, f)
            for f in allFiles
            if not f.startswith(".") and os.path.isfile(os.path.join(self.FOLDER, f))
        ]
        random.shuffle(videos)
        return videos

    def _loadVideoAtIndex(self, index):
        """Load and configure the reference video at *index* in the queue."""
        if not self.videoQueue:
            return

        # Cancel any in-progress annotation worker
        if hasattr(self, "_annotationWorker") and self._annotationWorker is not None:
            self._annotationWorker.finished.disconnect()
            self._annotationWorker.quit()
            self._annotationWorker.wait()
            self._annotationWorker = None

        path = self.videoQueue[index]

        if hasattr(self, "referenceVideo") and self.referenceVideo is not None:
            self.referenceVideo.release()
        if hasattr(self, "annotatedReferenceVideo") and self.annotatedReferenceVideo is not None:
            self.annotatedReferenceVideo.release()

        nonAnnotatedVideo = cv2.VideoCapture(path)
        self.referenceAnnotation = HandAnnotation(nonAnnotatedVideo)

        filename = os.path.basename(path)
        basename, ext = os.path.splitext(filename)
        annotatedPath = os.path.join(self.ANNOTATED_FOLDER, f"{basename}_annotated{ext}")

        if os.path.isfile(annotatedPath):
            # Already annotated — load landmarks from cache and continue immediately
            self.annotateReferenceVideo(path)
            self._finishLoading(path, annotatedPath)
        else:
            # Need to annotate — show loading screen and offload to background thread
            self._pendingPath = path
            self._pendingAnnotatedPath = annotatedPath
            if hasattr(self, "refStack"):
                self.refStack.setCurrentIndex(1)
            if hasattr(self, "refTimer"):
                self.refTimer.stop()
            self.isTracking = False
            self.userLandmarksTimestamped = []
            for b in (self.btnPrev, self.btnPlayPause, self.btnNext):
                b.setEnabled(False)
            if hasattr(self, "refProgressBar"):
                self.refProgressBar.reset()
            self._annotationWorker = AnnotationWorker(self.referenceAnnotation, path, annotatedPath)
            self._annotationWorker.finished.connect(self._onAnnotationFinished)
            self._annotationWorker.start()
            print(f"{BLUE}Annotating '{path}' in background…{ENDC}")

    def _onAnnotationFinished(self):
        """Called on the main thread when the background annotation worker completes."""
        self._annotationWorker = None
        self.annotateReferenceVideo(self._pendingPath)  # loads landmarks from the now-existing file
        self._finishLoading(self._pendingPath, self._pendingAnnotatedPath)
        if hasattr(self, "refStack"):
            self.refStack.setCurrentIndex(0)
        if hasattr(self, "refTimer"):
            self.refTimer.start(int(1000 / BASE_FPS))
        for b in (self.btnPrev, self.btnPlayPause, self.btnNext):
            b.setEnabled(True)
        print(f"{GREEN}✓{ENDC} Annotation complete, starting playback.")

    def _finishLoading(self, path, annotatedPath):
        """Open captures, reset state, and wire up scoring after a video is ready."""
        self.referenceVideo = cv2.VideoCapture(path)
        self.annotatedReferenceVideo = cv2.VideoCapture(annotatedPath) if os.path.isfile(annotatedPath) else None
        self.referenceVideoPauseUntil = 0.0
        self.isTracking = False
        self.userLandmarksTimestamped = []

        # Read pre-computed marker positions (0-1000 scale) from annotation object
        markerStart = getattr(self.referenceAnnotation, "markerStart", 0)
        markerEnd = getattr(self.referenceAnnotation, "markerEnd", 1000)

        # Seek both captures to the beginning
        self.referenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.annotatedReferenceVideo is not None:
            self.annotatedReferenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if hasattr(self, "refProgressBar"):
            self.refProgressBar.setValue(0)
            self.refProgressBar.setTrimMarkers(markerStart, markerEnd)

        if hasattr(self, "webcamAnnotation"):
            self.scoring = Scoring(self.webcamAnnotation, self.referenceAnnotation)

        print(f"{GREEN}✓{ENDC} Loaded reference video [{self.videoQueueIndex}]: '{path}'.")

    def loadReferenceVideo(self):
        """Build queue and load the first reference video.

        Returns:
            Path to the loaded reference video file, or None on error."""

        self.videoQueue = self._buildVideoQueue()
        if not self.videoQueue:
            return None

        self.videoQueueIndex = 0
        self._loadVideoAtIndex(self.videoQueueIndex)
        return self.videoQueue[self.videoQueueIndex]

    def setupPlayer(self):
        """Initialize reference video playback and start the display refresh timer."""
        self.loadReferenceVideo()

        # Setup a separate timer for the reference video.
        self.refTimer = QtCore.QTimer()
        self.refTimer.timeout.connect(self.displayReferenceVideo)
        self.refTimer.start(int(1000 / BASE_FPS))

        print(f"{GREEN}✓{ENDC} Player setup complete.")

    def saveUserLandmarks(self):
        """Save the collected webcam landmark sequence to a JSON file."""
        if not self.userLandmarksTimestamped:
            print(f"{WARNING}Warning{ENDC}: No user landmarks recorded, skipping save.")
            return

        os.makedirs(self.ANNOTATED_FOLDER, exist_ok=True)
        outFile = os.path.join(self.ANNOTATED_FOLDER, "webcam_userLandmarks.json")

        userData = []
        for timestamp, landmarks in self.userLandmarksTimestamped:
            userData.append({"timestamp": round(timestamp, 3), "landmarks": landmarks})

        with open(outFile, "w") as f:
            json.dump(userData, f, indent=2)

        print(f"{GREEN}✓{ENDC} Saved user landmarks to '{outFile}'.")

    def displayReferenceVideo(self):
        """Display the next frame from the reference video, looping from the start when finished.

        Respects pause state, loop-pause delay, and annotation visibility toggle.
        """
        if self.refPaused:
            return

        now = time.time()
        if now < self.referenceVideoPauseUntil:
            return

        if not hasattr(self, "referenceVideo") or self.referenceVideo is None:
            return

        ret, frame = self.referenceVideo.read()
        # Read the annotated frame in sync (discard if not needed)
        if self.annotatedReferenceVideo is not None:
            retA, annotatedFrame = self.annotatedReferenceVideo.read()
        else:
            retA, annotatedFrame = False, None

        currentFrame = int(self.referenceVideo.get(cv2.CAP_PROP_POS_FRAMES))

        if not ret:
            self.isTracking = False  # Stop tracking when video ends
            print(f"{HEADER}Stop tracking.{ENDC}")
            self.updateScore()  # Update the score display when the reference video finishes
            self.saveUserLandmarks()  # Save collected webcam landmarks to file

            self.referenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.annotatedReferenceVideo is not None:
                self.annotatedReferenceVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.referenceVideoPauseUntil = time.time() + self.referenceVideoLoopPause
            return

        if ret:
            if not self.isTracking:
                # Start tracking when the reference video begins playing
                self.isTracking = True
                print(f"{HEADER}Start tracking.{ENDC}")

                self.startTime = time.time()
                self.nextSampleTime = 0.0
                self.userLandmarksTimestamped = []

            if self.showAnnotations and retA and annotatedFrame is not None:
                rgbFrame = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
                image = self.referenceAnnotation.convertFrameToQtImage(rgbFrame)
            else:
                rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbFrame.shape
                bytesPerLine = ch * w
                image = QtGui.QImage(rgbFrame.data, w, h, bytesPerLine, QtGui.QImage.Format.Format_RGB888)

            self.gestureVideo.setPixmap(QtGui.QPixmap.fromImage(image))

            # Update progress bar — map currentFrame to the 0-1000 scale
            totalFrames = int(self.referenceVideo.get(cv2.CAP_PROP_FRAME_COUNT))
            if totalFrames > 1:
                self.refProgressBar.setValue(int((currentFrame - 1) / (totalFrames - 1) * 1000))

    def updateScore(self):
        """Update the score display label with the current score, color-coded by performance level."""

        currentScore = self.scoring.calculateScore(
            self.userLandmarksTimestamped
        )  # Calculate the score based on user landmarks during tracking
        # print(f"Number of user landmarks: {len(self.userLandmarksTimestamped)}")
        pct = currentScore * 100
        self.score.setText(f"Score: {pct:.1f}%")

        if pct >= 70:
            color = "#4caf50"  # green
        elif pct >= 40:
            color = "#ff9800"  # orange
        else:
            color = "#f44336"  # red

        self.score.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {color};"
            "background-color: #2a2a3e; border-radius: 8px; padding: 8px 16px;"
        )

        # print(f"Raw score: {currentScore:.4f} ({pct:.1f}%)")


if __name__ == "__main__":
    # Initialize and run the Qt application
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    print(f"{GREEN}✓{ENDC} Application initialized successfully.")

    sys.exit(app.exec())
