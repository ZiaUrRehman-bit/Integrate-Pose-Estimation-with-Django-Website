import cv2
import mediapipe as mp
import time
import numpy as np

class poseDetector():

    def __init__(self, mode = False, modelComp = 1, smoothLm = True,
                 enalbeSeg = False, smoothSeg = True, detectionCon = 0.5,
                 trackCon = 0.5):
        self.mode = mode
        self.modelComp = modelComp
        self.smoothLm = smoothLm
        self.enalbeSeg = enalbeSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smoothLm,
                                     self.enalbeSeg, self.smoothSeg,
                                     self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
            cv2.putText(frame, "zia", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        4, (255,0,255), 3)
        return frame

    def findPosLm(self, frame):

        self.poseLm = []

        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = frame.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.poseLm.append([id, cx, cy])
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        return self.poseLm

    def findAngle(self, x, y, z):

        x = np.array(x)  # First
        y = np.array(y)  # Mid
        z = np.array(z)  # End

        radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
        self.angle = np.abs(radians * 180.0 / np.pi)

        if self.angle > 180.0:
            self.angle = 360 - self.angle

        return self.angle





