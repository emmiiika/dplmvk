from HandAnnotation import HandAnnotation
import cv2


class Scoring:
    def __init__(self, referenceVideo, userVideo):
        self.referenceVideo = referenceVideo
        self.userVideo = userVideo

    def calculateScore(self, annotation):

        handLandmarks = annotation.getHandLandmarks()
        print(handLandmarks)

        # Placeholder for actual scoring logic
        score = 0.0
        # Implement scoring algorithm here
        return score
