from HandAnnotation import HandAnnotation
import cv2


class Scoring:
    def __init__(self, webcamAnnotation, referenceAnnotation):
        self.webcamAnnotation = webcamAnnotation
        self.referenceAnnotation = referenceAnnotation

    def calculateScore(self):

        webcamHandLandmarks = self.webcamAnnotation.getHandLandmarks()
        referenceHandLandmarks = self.referenceAnnotation.getHandLandmarks()
        # print(webcamHandLandmarks)
        # print(referenceHandLandmarks)

        # Placeholder for actual scoring logic
        score = 0.0
        # Implement scoring algorithm here
        return score
