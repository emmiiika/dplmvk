import cv2
import numpy as np
import mediapipe as mp
from mediapipe import framework
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia
import os
import copy


# ANSI escape codes for colored terminal output
HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"


class HandAnnotation:
    """Handles hand landmark detection, visualization, and video recording using MediaPipe and OpenCV."""

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    SAMPLINGRATE = 0.05  # seconds (20 FPS)

    def __init__(self, videoInput):
        """
        Initialize the hand detector and video writer.

        Args:
            videoInput: OpenCV VideoCapture object for accessing the video feed.

        Note:
            This creates an 'output.mp4' file in the current directory for recording.
        """
        self.cam = videoInput

        # Get video properties for recording
        frameWidth = int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video codec and output file writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        self.out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frameWidth, frameHeight))

        # Configure MediaPipe HandLandmarker
        baseOptions = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=baseOptions, num_hands=2)

        # Load the pre-trained hand landmarking model
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.handLandmarksList = []  # Store detected hand landmarks for external access
        self.handLandmarksTimestamped = []  # Store timestamped hand landmarks

    def getHandLandmarks(self):
        """Return the list of detected hand landmarks from the most recent detection result."""
        return self.handLandmarksList

    def extractHandLandmarkProtos(self, detectionResult):
        """Convert MediaPipe detection landmarks into NormalizedLandmarkList protos for drawing."""
        handLandmarksList = detectionResult.hand_landmarks
        protos = []

        for handLandmarks in handLandmarksList:
            handLandmarksProto = framework.formats.landmark_pb2.NormalizedLandmarkList()
            handLandmarksProto.landmark.extend(
                [
                    framework.formats.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in handLandmarks
                ]
            )
            protos.append(handLandmarksProto)

        return protos

    def drawLandmarksOnImage(self, rgbImage, detectionResult):
        """Overlay detected hand landmarks, connections, and handedness labels onto the image."""
        self.handLandmarksList = detectionResult.hand_landmarks
        handednessList = detectionResult.handedness
        annotatedImage = np.copy(rgbImage)

        handProtos = self.extractHandLandmarkProtos(detectionResult)

        # Iterate through each detected hand
        for idx, handLandmarksProto in enumerate(handProtos):
            handedness = handednessList[idx]

            mp.solutions.drawing_utils.draw_landmarks(
                annotatedImage,
                handLandmarksProto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Calculate text position for handedness label (above the hand)
            height, width, _ = annotatedImage.shape
            handLandmarks = self.handLandmarksList[idx]
            xCoordinates = [landmark.x for landmark in handLandmarks]
            yCoordinates = [landmark.y for landmark in handLandmarks]
            textX = int(min(xCoordinates) * width)
            textY = int(min(yCoordinates) * height) - self.MARGIN

            # Draw "Left" or "Right" text label above the hand
            cv2.putText(
                annotatedImage,
                f"{handedness[0].category_name}",
                (textX, textY),
                cv2.FONT_HERSHEY_DUPLEX,
                self.FONT_SIZE,
                self.HANDEDNESS_TEXT_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotatedImage

    def convertFrameToQtImage(self, rgbFrame):
        """Convert an RGB OpenCV frame to Qt QImage format.

        Args:
            rgbFrame: RGB frame (numpy array) from OpenCV.

        Returns:
            QImage object in RGB format ready for Qt display.
        """
        return QtGui.QImage(  # type: ignore
            rgbFrame,
            rgbFrame.shape[1],
            rgbFrame.shape[0],
            rgbFrame.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )

    def processSpecificFrame(self, frame, returnQt=False):
        """Process a provided frame (instead of capturing from camera) for hand detection and annotation.

        Args:
            frame: Input BGR image (numpy array) to process.
            returnQt: If True, return a Qt QImage; otherwise, return BGR image.

        Returns:
            annotated_frame: BGR image with hand annotations if returnQt=False, or QImage if returnQt=True, or None if frame is invalid.
        """
        if frame is None:
            return None

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        detectionResult = self.detector.detect(image)
        annotatedImage = self.drawLandmarksOnImage(image.numpy_view(), detectionResult)

        if returnQt:
            return self.convertFrameToQtImage(annotatedImage)
        else:
            return cv2.cvtColor(annotatedImage, cv2.COLOR_RGB2BGR)

    def saveLandmarksToFile(self, outputPath):
        """Save the detected hand landmarks to a text file for later analysis.

        Args:
            outputPath: Path to the annotated video file (string).
        """
        # Preserve base path and replace extension with _handLandmarks.txt
        base, _ = os.path.splitext(outputPath)
        filePath = f"{base}_handLandmarks.txt"

        with open(filePath, "a") as f:
            for timestamp, handLandmarks in self.handLandmarksTimestamped:
                f.write(f"time={timestamp:.3f}\n")
                for hand in handLandmarks:
                    for landmark in hand:
                        f.write(f"{landmark.x}, {landmark.y}, {landmark.z}\n")
                f.write("\n\n")  # Separate hands with two blank lines

    def createAnnotatedVideo(self, videoPath, outputPath):
        """
        Create an annotated video from an input video file by processing each frame.

        Reads frames from the input video, applies hand detection and annotation,
        and writes the result to a new output video file.

        Args:
            videoPath: Path to the input video file (string).
            outputPath: Path for the output annotated video file (string).

        Returns:
            annotated_video: OpenCV VideoCapture object for the output video, or None on failure.
        """
        print(f"Creating annotated video from {HEADER}'{videoPath}'{ENDC} at '{HEADER}{outputPath}{ENDC}'.")
        video = cv2.VideoCapture(videoPath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frameIdx = 0

        # Initialize VideoWriter with MP4 codec at original FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        if not video.isOpened():
            print(f"{FAIL}Error{ENDC}: Could not open video at '{videoPath}'.")
            return None
        print(f"{GREEN}✓{ENDC} Video opened successfully from '{videoPath}'.")

        if not out.isOpened():
            print(f"{FAIL}Error{ENDC}: Could not initialize VideoWriter.")
            video.release()
            return None
        print(f"{GREEN}✓{ENDC} VideoWriter initialized successfully.")

        framesProcessed = 0
        self.handLandmarksTimestamped = []  # Reset for new video
        nextSampleTime = 0.0  # Sample at 20 FPS (every 0.05 seconds)
        ret, frame = video.read()
        while ret:
            annotated = self.processSpecificFrame(frame)

            if annotated is not None:
                out.write(annotated)
                framesProcessed += 1

            timestampSeconds = frameIdx / fps
            if timestampSeconds >= nextSampleTime:
                self.handLandmarksTimestamped.append([timestampSeconds, copy.deepcopy(self.handLandmarksList)])
                nextSampleTime += self.SAMPLINGRATE

            frameIdx += 1

            ret, frame = video.read()

        video.release()
        out.release()

        self.saveLandmarksToFile(outputPath)  # Save landmarks to a text file alongside the video

        self.annotatedVideo = cv2.VideoCapture(outputPath)

        print(f"{GREEN}✓{ENDC} Annotated video created with {framesProcessed} processed frames.")
        return self.annotatedVideo
