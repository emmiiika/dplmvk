import cv2
import numpy as np
import mediapipe as mp
from mediapipe import framework


class HandAnnotation:

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    def __init__(self, cam):
        # ================================ WEBCAM =====================================
        # Open the default camera
        # cam = cv2.VideoCapture(0)
        self.cam = cam

        # Get the default frame width and height
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        self.out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))
        # ================================ WEBCAM =====================================

        # =============================== RECORDING ===================================
        # # Open the video file
        # video_path = "../videá/dom - Snepeda (360p, h264).mp4"
        # cam = cv2.VideoCapture(video_path)

        # if not cam.isOpened():
        #     print("Error: Could not open video file.")
        #     exit()

        # # Get properties from the source video for the output writer
        # frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cam.get(cv2.CAP_PROP_FPS)  # Get original FPS

        # # 3. Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter("output_processed.mp4", fourcc, fps, (frame_width, frame_height))
        # =============================== RECORDING ===================================

        # Create HandLandmarker object once, before loop
        # Path to the pre-trained AI model file
        base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
        # Configure AI behavior (model link + limit to 2 hands)
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        # Initialize detector; loads model into memory
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = framework.formats.landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    framework.formats.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
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
        # Release the capture and writer objects
        self.out.release()

    def processFrame(self):
        ret, frame = self.cam.read()

        if not ret:
            return None

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to Mediapipe Image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        detection_result = self.detector.detect(image)

        # Draw landmarks
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        # cv2.imshow("Camera", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Write to output file
        self.out.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # # Release the capture and writer objects
        # cam.release()
        # cv2.destroyAllWindows()
