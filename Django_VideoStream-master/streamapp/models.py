from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
from numpy import savetxt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class poseDetector():
    def __init__(self, mode=False, modelComp = 1, smooth=True, enable_seg=False,
                    smooth_seg=False, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComp = modelComp
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smooth, self.enable_seg, self.smooth_seg,
                                        self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                #self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                #                            self.mpPose.POSE_CONNECTIONS)
                IsPose = True
        else:
                IsPose = False
        return img, IsPose

    def findPosition(self, img, draw=True):
        self.lmList = []
        self.world = self.results.pose_world_landmarks.landmark
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                #if draw:
                #    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return self.lmList, self.world

    def findAngle(self, img, p1, p2, p3, draw=False):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        joint1 = np.array([self.world[p1].x, self.world[p1].y, self.world[p1].z])
        joint2 = np.array([self.world[p2].x, self.world[p2].y, self.world[p2].z])
        joint3 = np.array([self.world[p3].x, self.world[p3].y, self.world[p3].z])

        # Calculate the Angle
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(joint1-joint2, joint3-joint2)),
                                            np.dot(joint1-joint2, joint3- joint2)))

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            #cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
            #            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle

    def angle2d(self, img, p1, p2, p3, draw=False):
        # Get the landmarks
        xy1 = np.array(self.lmList[p1][1:])
        xy2 = np.array(self.lmList[p2][1:])
        xy3 = np.array(self.lmList[p3][1:])


        # Calculate the Angle
        angle = np.rad2deg(np.arctan2(np.linalg.det([xy1-xy2, xy3-xy2]), np.dot(xy1-xy2, xy3- xy2)))

        # print(angle)

        # Draw
        if draw:
            cv2.putText(img, str(int(angle)), xy2,
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            # cv2.line(img,xy1,xy2,(0, 255, 255), 3)
            # cv2.line(img,(xy3[0],0),xy2,(0, 255, 255), 3)
            # cv2.circle(img,xy2,5,(255,0,255),cv2.FILLED)


        return np.abs(angle)

