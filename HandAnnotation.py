import cv2
import numpy as np
import mediapipe as mp  # type: ignore
from mediapipe import framework
from PySide6 import QtGui
import os
import json
from LandmarkIndices import LandmarkIndices


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
    SAMPLING_RATE = 1.0 / 30.0  # seconds (30 FPS)
    TRIM_PADDING_SECONDS = 0.25  # seconds of padding kept before/after active hand region
    MOVEMENT_THRESHOLD = 0.01  # mean landmark displacement (normalised 0-1 image coords) to count as movement

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

        self.markerStart = 0  # progress-bar position of movement start (0-1000 scale)
        self.markerEnd = 1000  # progress-bar position of movement end (0-1000 scale)

        self.handLandmarksList = []  # Store detected hand landmarks for external access
        self.handLandmarksTimestamped = []  # Store timestamped hand landmarks
        self.wristTrajectoryList = []  # Store original wrist positions for each hand and frame
        self.currentWristPositions = []  # Original wrist [x,y,z] per hand for the latest frame

    # ------------------------------------------------------------------
    # Landmark processing helpers
    # ------------------------------------------------------------------

    def _getTranslatedLandmarks(self, handLandmarks):
        """Return wrist-centered landmark numpy array for one hand.

        Args:
            handLandmarks: List of Landmark objects for one hand.
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks])
        if coords.shape[0] == 0:
            return coords
        wrist = coords[LandmarkIndices.WRIST]  # Use wrist as origin of the hand coordinate system
        translatedCoords = coords - wrist
        return translatedCoords

    def _getNormalizedScaleLandmarks(self, translatedCoords):
        """Normalize landmarks by hand size to be invariant to distance from camera.

        Uses the distance from wrist to middle finger MCP (middle knuckle) as the scale reference.
        This is robust and consistent across different hand sizes and distances.

        Args:
            translatedCoords: Numpy array of wrist-centered landmarks (21 x 3).

        Returns:
            Numpy array of scale-normalized landmarks, or original if scale is 0.
        """
        wrist = translatedCoords[LandmarkIndices.WRIST]
        middleMCP = translatedCoords[LandmarkIndices.MIDDLE_MCP]
        scale = np.linalg.norm(middleMCP - wrist)  # Euclidean distance from wrist to middle MCP as hand size reference

        # Avoid division by zero if hand is not detected or scale is extremely small
        if scale == 0:
            return translatedCoords

        normalized_coords = translatedCoords / scale
        return normalized_coords

    def _landmarksToDict(self, handLandmarksList):
        """Convert hand landmarks to dictionary format (x, y, z coordinates only).

        Args:
            handLandmarksList: List of lists of Landmark objects or NormalizedLandmark objects.

        Returns:
            List of lists of dictionaries with x, y, z keys.
        """
        handLandmarksDict = []
        for hand in handLandmarksList:
            handData = [
                {"x": float(landmark[0]), "y": float(landmark[1]), "z": float(landmark[2])} for landmark in hand
            ]
            handLandmarksDict.append(handData)
        return handLandmarksDict

    # ------------------------------------------------------------------
    # Landmark extraction and drawing
    # ------------------------------------------------------------------

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
        # Store original wrist positions before normalization
        self.wristTrajectoryList = []
        original_coords = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in detectionResult.hand_landmarks]
        self.currentWristPositions = []
        for coords in original_coords:
            if coords.shape[0] > LandmarkIndices.WRIST:
                wrist = coords[LandmarkIndices.WRIST]
                self.wristTrajectoryList.append(wrist.tolist())
                self.currentWristPositions.append(wrist.tolist())
            else:
                self.wristTrajectoryList.append([None, None, None])
                self.currentWristPositions.append(None)

        # Normalize landmarks: translate (wrist-centered) then scale (hand-size invariant)
        translated = [self._getTranslatedLandmarks(hand) for hand in detectionResult.hand_landmarks]
        normalized = [self._getNormalizedScaleLandmarks(trans) for trans in translated]
        self.handLandmarksList = self._landmarksToDict(normalized)
        annotatedImage = np.copy(rgbImage)

        handProtos = self.extractHandLandmarkProtos(detectionResult)

        for handLandmarksProto in handProtos:
            mp.solutions.drawing_utils.draw_landmarks(
                annotatedImage,
                handLandmarksProto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

        return annotatedImage

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

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

    def processSpecificFrame(self, frame, returnQt=False, drawAnnotations=True):
        """Process a provided frame (instead of capturing from camera) for hand detection and annotation.

        Args:
            frame: Input BGR image (numpy array) to process.
            returnQt: If True, return a Qt QImage; otherwise, return BGR image.
            drawAnnotations: If True, draw landmarks and labels on output frame.

        Returns:
            annotated_frame: BGR image with hand annotations if returnQt=False, or QImage if returnQt=True, or None if frame is invalid.
        """
        if frame is None:
            return None

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        detectionResult = self.detector.detect(image)

        # Always compute and store normalized landmarks for downstream scoring.
        translated = [self._getTranslatedLandmarks(hand) for hand in detectionResult.hand_landmarks]
        normalized = [self._getNormalizedScaleLandmarks(trans) for trans in translated]
        self.handLandmarksList = self._landmarksToDict(normalized)

        # Always update raw wrist positions so scoring has them even when drawing is off.
        original_coords = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in detectionResult.hand_landmarks]
        self.currentWristPositions = []
        for coords in original_coords:
            if coords.shape[0] > LandmarkIndices.WRIST:
                self.currentWristPositions.append(coords[LandmarkIndices.WRIST].tolist())
            else:
                self.currentWristPositions.append(None)

        if drawAnnotations:
            outputImage = self.drawLandmarksOnImage(image.numpy_view(), detectionResult)
        else:
            outputImage = image.numpy_view()

        if returnQt:
            return self.convertFrameToQtImage(outputImage)
        else:
            return cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Landmark serialization
    # ------------------------------------------------------------------

    def saveLandmarksToFile(self, outputPath):
        """Save the detected hand landmarks to a JSON file for later analysis and easy loading.

        Args:
            outputPath: Path to the annotated video file (string).
        """
        # Preserve base path and replace extension with _handLandmarks.json
        base, _ = os.path.splitext(outputPath)
        filePath = f"{base}_handLandmarks.json"

        # Convert landmarks to a serializable format
        landmarksData = []  # type: ignore
        for entry in self.handLandmarksTimestamped:
            timestamp = entry[0]
            handLandmarks = entry[1]
            # Flatten all hands into one list of landmark dicts
            allLandmarks = []
            for hand in handLandmarks:
                allLandmarks.extend(hand)
            frameData = {"timestamp": round(timestamp, 3), "landmarks": allLandmarks}
            # Include raw wrist positions if saved alongside landmarks
            if len(entry) > 2 and entry[2] is not None:
                frameData["wrist"] = entry[2]
            landmarksData.append(frameData)
        with open(filePath, "w") as f:
            json.dump(
                {"markerStart": self.markerStart, "markerEnd": self.markerEnd, "frames": landmarksData}, f, indent=2
            )
        print(f"{GREEN}✓{ENDC} Landmarks saved to '{filePath}'")

    def loadLandmarksFromFile(self, filePath):
        """Load timestamped hand landmarks from a JSON file.

        Args:
            filePath: Path to the JSON landmark file.

        Returns:
            List of tuples (timestamp, handLandmarks) or None if file doesn't exist.
        """
        if not os.path.exists(filePath):
            print(f"{FAIL}Error{ENDC}: Landmark file not found at '{filePath}'.")
            return None

        try:
            with open(filePath, "r") as f:
                data = json.load(f)

            # Support dict format {"markerStart":…, "markerEnd":…, "frames":[…]} and old plain-list format
            if isinstance(data, dict):
                landmarksData = data.get("frames", [])
                self.markerStart = data.get("markerStart", 0)
                self.markerEnd = data.get("markerEnd", 1000)
            else:
                landmarksData = data
                self.markerStart = 0
                self.markerEnd = 1000

            # Convert back to (timestamp, handLandmarks) tuples
            self.handLandmarksTimestamped = []
            for frameData in landmarksData:
                timestamp = frameData["timestamp"]
                # Flatten landmarks are stored as one hand
                hands = [frameData["landmarks"]]
                wrist = frameData.get("wrist", None)
                self.handLandmarksTimestamped.append((timestamp, hands, wrist))

            print(f"{GREEN}✓{ENDC} Loaded {len(self.handLandmarksTimestamped)} frames from '{filePath}'")
            return self.handLandmarksTimestamped
        except json.JSONDecodeError as e:
            print(f"{FAIL}Error{ENDC}: Failed to parse JSON file: {e}")
            return None
        except Exception as e:
            print(f"{FAIL}Error{ENDC}: Failed to load landmarks: {e}")
            return None

    # ------------------------------------------------------------------
    # Video annotation
    # ------------------------------------------------------------------

    def _computeTrimMarkers(self, landmarksPerFrame, totalFrames, fps):
        """Compute trim frame indices from pre-collected per-frame 2-D landmark data.

        Args:
            landmarksPerFrame: List of (frameIdx, coords_or_None) tuples where coords is
                               a (N, 2) numpy array of image-space landmark positions.
            totalFrames: Total number of frames in the video.
            fps: Frames per second (used to compute padding in frames).

        Returns:
            (trimStart, trimEnd) frame indices with padding applied, or (0, totalFrames-1)
            if no movement is detected.
        """
        if not landmarksPerFrame:
            return 0, max(totalFrames - 1, 0)

        firstActive = None
        lastActive = None
        prevCoords = None

        for frameIdx, coords in landmarksPerFrame:
            if coords is None:
                prevCoords = None
                continue
            if prevCoords is not None and coords.shape == prevCoords.shape:
                displacement = np.mean(np.linalg.norm(coords - prevCoords, axis=1))
                if displacement >= self.MOVEMENT_THRESHOLD:
                    if firstActive is None:
                        firstActive = frameIdx
                    lastActive = frameIdx
            prevCoords = coords

        if firstActive is None:
            print(f"{WARNING}No hand movement detected, skipping trim.{ENDC}")
            return 0, max(totalFrames - 1, 0)

        pad = int(fps * self.TRIM_PADDING_SECONDS)
        trimStart = max(0, firstActive - pad if firstActive is not None else 0)
        if lastActive is not None:
            trimEnd = min(totalFrames - 1, lastActive + pad)
        else:
            trimEnd = totalFrames - 1
        trimmedSeconds = (trimEnd - trimStart + 1) / fps
        print(
            f"{BLUE}Trim range: frames {trimStart}–{trimEnd} "
            f"({trimmedSeconds:.1f}s / {totalFrames / fps:.1f}s total){ENDC}"
        )
        return trimStart, trimEnd

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
        if not video.isOpened():
            print(f"{FAIL}Error{ENDC}: Could not open video at '{videoPath}'.")
            return None

        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"{FAIL}Error{ENDC}: Could not initialize VideoWriter.")
            video.release()
            return None
        print(f"{GREEN}✓{ENDC} VideoWriter initialized successfully.")

        framesProcessed = 0
        self.handLandmarksTimestamped = []  # Reset for new video
        nextSampleTime = 0.0
        frameIdx = 0
        landmarksPerFrame = []  # collected for trim marker computation after the loop

        ret, frame = video.read()
        while ret:
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
            detectionResult = self.detector.detect(image)

            # Annotate frame — also sets self.handLandmarksList and self.currentWristPositions
            annotatedImage = self.drawLandmarksOnImage(image.numpy_view(), detectionResult)
            out.write(cv2.cvtColor(annotatedImage, cv2.COLOR_RGB2BGR))
            framesProcessed += 1

            timestampSeconds = frameIdx / fps
            if timestampSeconds >= nextSampleTime:
                self.handLandmarksTimestamped.append(
                    [timestampSeconds, self.handLandmarksList, list(self.currentWristPositions)]
                )
                nextSampleTime += self.SAMPLING_RATE

            # Collect 2-D positions for trim computation (reuses already-detected landmarks)
            if detectionResult.hand_landmarks:
                coords2d = np.array([[lm.x, lm.y] for hand in detectionResult.hand_landmarks for lm in hand])
                landmarksPerFrame.append((frameIdx, coords2d))
            else:
                landmarksPerFrame.append((frameIdx, None))  # type: ignore

            frameIdx += 1
            ret, frame = video.read()

        video.release()
        out.release()

        # Compute trim markers from detection data collected during the single pass above
        trimStart, trimEnd = self._computeTrimMarkers(landmarksPerFrame, totalFrames, fps)
        denom = max(1, totalFrames - 1)
        self.markerStart = int(trimStart / denom * 1000)
        self.markerEnd = int(trimEnd / denom * 1000)

        self.saveLandmarksToFile(outputPath)  # Save landmarks to a text file alongside the video

        self.annotatedVideo = cv2.VideoCapture(outputPath)

        print(f"{GREEN}✓{ENDC} Annotated video created with {framesProcessed} processed frames.")
        return self.annotatedVideo
