from HandAnnotation import HandAnnotation
import cv2


class Scoring:
    def __init__(self, webcamAnnotation, referenceAnnotation):
        self.webcamAnnotation = webcamAnnotation
        self.referenceAnnotation = referenceAnnotation

    def calculateScore(self, user_landmarks=None):
        # print(
        #     f"Calculating score, user_landmarks: {len(user_landmarks) if user_landmarks else 0}, reference_landmarks: {len(self.referenceAnnotation.handLandmarksTimestamped)}"
        # )

        if user_landmarks is None:
            # Fallback to current landmarks if no sequence provided
            webcamHandLandmarks = self.webcamAnnotation.getHandLandmarks()
            referenceHandLandmarks = self.referenceAnnotation.getHandLandmarks()
            # Placeholder for single-frame scoring
            score = 0.0
            return score

        # Compare timestamped sequences
        reference_landmarks = self.referenceAnnotation.handLandmarksTimestamped
        # Implement sequence comparison logic here
        # For now, placeholder
        score = 0.0

        # print(user_landmarks)
        # print(reference_landmarks)

        return score
