import cv2
import numpy as np
import mediapipe as mp
from mediapipe import framework
from PySide6 import QtCore, QtWidgets, QtGui, QtMultimediaWidgets, QtMultimedia


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

    def __init__(self, cam):
        """
        Initialize the hand detector and video writer.

        Args:
            cam: OpenCV VideoCapture object for accessing the camera feed.

        Note:
            This creates an 'output.mp4' file in the current directory for recording.
        """
        self.cam = cam

        # Get video properties for recording
        frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video codec and output file writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        self.out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frameWidth, frameHeight))

        # Configure MediaPipe HandLandmarker
        baseOptions = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=baseOptions, num_hands=2)

        # Load the pre-trained hand landmarking model
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def drawLandmarksOnImage(self, rgbImage, detectionResult):
        """Overlay detected hand landmarks, connections, and handedness labels onto the image.

        This method draws skeletal connections between landmarks, individual landmark points,
        and labels indicating left/right hand above each detected hand.

        Args:
            rgb_image: Original image in RGB format (numpy array).
            detection_result: MediaPipe HandLandmarker detection results containing landmarks and handedness.

        Returns:
            annotated_image: RGB image with visual overlays (numpy array).
        """
        handLandmarksList = detectionResult.hand_landmarks
        handednessList = detectionResult.handedness
        annotatedImage = np.copy(rgbImage)

        # Iterate through each detected hand
        for idx in range(len(handLandmarksList)):
            handLandmarks = handLandmarksList[idx]
            handedness = handednessList[idx]

            # Convert landmarks to a format drawing_utils can use
            handLandmarksProto = framework.formats.landmark_pb2.NormalizedLandmarkList()
            handLandmarksProto.landmark.extend(
                [
                    framework.formats.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in handLandmarks
                ]
            )

            # Draw skeletons and landmark points
            mp.solutions.drawing_utils.draw_landmarks(
                annotatedImage,
                handLandmarksProto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Calculate text position for handedness label (above the hand)
            height, width, _ = annotatedImage.shape
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

    def processSpecificFrame(self, frame):
        """Process a provided frame (instead of capturing from camera) for hand detection and annotation.

        Args:
            frame: Input BGR image (numpy array) to process.

        Returns:
            annotated_frame: BGR image with hand annotations, or None if frame is invalid.
        """
        if frame is None:
            return None

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        detectionResult = self.detector.detect(image)
        annotatedImage = self.drawLandmarksOnImage(image.numpy_view(), detectionResult)

        return cv2.cvtColor(annotatedImage, cv2.COLOR_RGB2BGR)

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

    def annotateFrame(self, frame):
        """Process a single frame with hand detection and convert to Qt-compatible image format.

        Args:
            frame: Input BGR frame from OpenCV.

        Returns:
            QImage object in RGB format ready for display, or None if frame is invalid.
        """
        if frame is None:
            return None

        annotatedFrame = self.processSpecificFrame(frame)

        # Convert OpenCV BGR format to RGB for Qt display
        rgbFrame = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
        return self.convertFrameToQtImage(rgbFrame)

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

        # Initialize VideoWriter with MP4 codec at original FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        if not video.isOpened():
            print(f"{FAIL}Error: Could not open video at '{videoPath}'{ENDC}")
            return None
        print(f"{GREEN}✓ Video opened successfully from '{videoPath}'{ENDC}")

        if not out.isOpened():
            print(f"{FAIL}Error: Could not initialize VideoWriter.{ENDC}")
            video.release()
            return None
        print(f"{GREEN}✓ VideoWriter initialized successfully.{ENDC}")

        framesProcessed = 0
        ret, frame = video.read()
        while ret:
            annotated = self.processSpecificFrame(frame)

            if annotated is not None:
                out.write(annotated)
                framesProcessed += 1

            ret, frame = video.read()

        video.release()
        out.release()

        self.annotatedVideo = cv2.VideoCapture(outputPath)

        print(f"{BLUE}✓ Annotated video created with {ENDC}{framesProcessed}{BLUE} processed frames.{ENDC}")
        return self.annotatedVideo
