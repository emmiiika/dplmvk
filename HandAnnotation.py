import cv2
import numpy as np
import mediapipe as mp
from mediapipe import framework

HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"


class HandAnnotation:
    """Handles hand landmark detection, visualization, and video recording."""

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    def __init__(self, cam):
        """
        Initialize the hand detector and video writer.

        Args:
            cam: OpenCV VideoCapture object.
        """
        self.cam = cam

        # Get video properties for recording
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video codec and output file writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        self.out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

        # Configure MediaPipe HandLandmarker
        base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

        # Load the pre-trained hand landmarking model
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def drawLandmarksOnImage(self, rgb_image, detection_result):
        """
        Overlay detected landmarks and handedness labels onto the image.

        Args:
            rgb_image: Original image in RGB format.
            detection_result: Data from MediaPipe detector.

        Returns:
            annotated_image: Image with visual feedback.
        """
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Iterate through each detected hand
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Convert landmarks to a format drawing_utils can use
            hand_landmarks_proto = framework.formats.landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    framework.formats.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )

            # Draw skeletons and points
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Calculate text position for handedness label (above the hand)
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw "Left" or "Right" text on the frame
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                self.FONT_SIZE,
                self.HANDEDNESS_TEXT_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def saveVideo(self):
        """Release the VideoWriter resource and return the writer object."""
        recording = self.out
        self.out.release()
        return recording

    def processFrame(self):
        """
        Capture a single frame, run AI detection, and update the video file.

        Returns:
            annotated_frame: BGR image ready for display or None if capture fails.
        """
        ret, frame = self.cam.read()

        if not ret:
            return None

        # MediaPipe requires RGB format, OpenCV uses BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap frame into MediaPipe Image object
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform the actual hand tracking
        detection_result = self.detector.detect(image)

        # Draw visual markers on the frame
        annotated_image = self.drawLandmarksOnImage(image.numpy_view(), detection_result)

        # Convert back to BGR to save and display via OpenCV/Qt
        bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        self.out.write(bgr_annotated)

        return bgr_annotated

    def processSpecificFrame(self, frame):
        """Processes a given frame instead of reading from self.cam."""
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(image)
        annotated_image = self.drawLandmarksOnImage(image.numpy_view(), detection_result)

        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    def createAnnotatedVideo(self, videoPath, outputPath):
        print(f"Creating annotated video from {HEADER}'{videoPath}'{ENDC} at '{HEADER}{outputPath}{ENDC}'.")
        video = cv2.VideoCapture(videoPath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        if not video.isOpened():
            print(f"{FAIL}Could not open video at '{videoPath}'{ENDC}")

            return None
        else:
            print(f"{GREEN}Video opened successfully from '{videoPath}'{ENDC}")

        if not out.isOpened():
            print(f"{FAIL}Could not open VideoWriter.{ENDC}")

            exit(1)
            return None
        else:
            print(f"{GREEN}VideoWriter opened successfully.{ENDC}")

        framesProcessed = 0
        ret, frame = video.read()
        while ret:
            annotated = self.processSpecificFrame(frame)
            out.write(annotated)

            if annotated is not None:
                out.write(annotated)
                framesProcessed += 1

            ret, frame = video.read()

        video.release()
        out.release()

        self.annotatedVideo = cv2.VideoCapture(outputPath)

        print(f"{BLUE}Created annotated video with {ENDC}{framesProcessed}{BLUE} processed frames.{ENDC}")
        return self.annotatedVideo
