
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import streamapp.PoseModule as pd
import streamapp.models as ps
import time

# posedec = pd.poseDetector()



class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	@property
	def get_frame(self):

		detector = ps.poseDetector()
		calib = [0, 0, 0, 0]
		calibIdx = 0
		disKnee = []
		calibNose = [320, 100]
		calibRHeel = [290, 420]
		calibLHeel = [340, 420]
		calbRes = 35
		status = 0
		countFrame = 0
		repDown = 0
		repUp = 1
		rep = 0

		# arrowLR = cv2.imread("streamapp\\arrowLR_no_bg.png")  # =>
		# arrowRL = cv2.imread("streamapp\\arrowRL_no_bg.png")  # <=
		# arrowUp = cv2.imread("streamapp\\arrowUp.png")
		# arrowLR = cv2.resize(arrowLR, (30, 30))
		# arrowRL = cv2.resize(arrowRL, (30, 30))
		# arrowUp = cv2.resize(arrowUp, (30, 30))

		arrowLR = "Right"
		arrowRL = "Left"
		arrowUp = "Up"

		success, img = self.video.read()
		# image = posedec.findPose(img)
		img = cv2.flip(img, 1)
		img, IsPose = detector.findPose(img, draw=True)
		frame_width = int(self.video.get(3))  # 640
		frame_height = int(self.video.get(4))  # 480

		if IsPose:
			lmList, world = detector.findPosition(img, draw=True)

			# X = []
			# Y = []
			# Z = []
			# for i in range(len(world)):
			#     X.append(world[i].x)
			#     Y.append(world[i].y)
			#     Z.append(world[i].z)

			# ax.set_xlim3d(-0.7,0.7)
			# ax.set_ylim3d(-0.7,0.7)
			# ax.set_zlim3d(-0.7,0.7)
			# ax.set_xlabel("X")
			# ax.set_ylabel("Y")
			# ax.scatter3D(np.array(X), np.array(Z), np.array(Y)*-1, color="green")
			# ax.plot3D((X[12],X[24]),(Z[12],Z[24]),(Y[12]*-1,Y[24]*-1),"red")
			# ax.plot3D((X[11],X[23]),(Z[11],Z[23]),(Y[11]*-1,Y[23]*-1),"blue")
			# fig.canvas.draw()
			# imgFig = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep = '')
			# imgFig = imgFig.reshape(fig.canvas.get_width_height()[::-1]+(3,))
			# imgFig = cv2.resize(imgFig,(200,200))

			# imgFig = cv2.cvtColor(imgFig,cv2.COLOR_RGB2BGR)

			# ax.cla()

			# img[280:480,0:200,:] = imgFig

			# cv2.imshow("Plot",imgFig)

			# ax = fig.add_subplot(projection='3d')

			# ax.scatter(1, 1, 1, color="green")
			# ax.cla()
			# fig.show()

			if len(lmList) != 0:
				## Joint angle of interest
				# Left Shoulder
				RightShoulderAng = detector.findAngle(img, 13, 11, 23)
				# Left Elbow
				RightElbowAng = detector.findAngle(img, 15, 13, 11)
				# Left Hip
				RightHipAng = detector.angle2d(img, 11, 23, 25, False)
				# Left Knee
				RightKneeAng = detector.angle2d(img, 23, 25, 27, False)
				# Left back
				RightbackAng = detector.angle2d(img, 7, 11, 23, False)

				# Right Shoulder
				LeftShoulderAng = detector.findAngle(img, 14, 12, 24)
				# Right Elbow
				LeftElbowAng = detector.findAngle(img, 16, 14, 12)
				# Right Hip
				LeftHipAng = detector.angle2d(img, 12, 24, 26, False)
				# Right Knee
				LeftKneeAng = detector.angle2d(img, 24, 26, 28, False)
				# Right back
				LeftbackAng = detector.angle2d(img, 8, 12, 24, False)

			if status == 0:
				xRShoulder, yRShoulder = lmList[24][1:]
				xLShoulder, yLShoulder = lmList[23][1:]
				xRHip, yRHip = lmList[26][1:]
				xLHip, yLHip = lmList[25][1:]
				xRKnee, yRKnee = lmList[26][1:]
				xLKnee, yLKnee = lmList[25][1:]
				xRToe, yRToe = lmList[32][1:]
				xLToe, yLToe = lmList[31][1:]

				cv2.line(img, (xLShoulder, 450), (xLShoulder + 50, 450), (255, 0, 0), 3)

				if np.abs(xRShoulder - xRHip) <= 25:
					calib[0] = 1
					cv2.line(img, (xRShoulder, 450), (xRShoulder - 50, 450), (255, 0, 0), 3)
				else:
					calib[0] = 0
					cv2.line(img, (xRShoulder, 450), (xRShoulder - 50, 450), (0, 255, 0), 3)

				if np.abs(xLShoulder - xLHip) <= 25:
					calib[1] = 1
					cv2.line(img, (xLShoulder, 450), (xLShoulder - 50, 450), (255, 0, 0), 3)
				else:
					calib[1] = 0
					cv2.line(img, (xLShoulder, 450), (xLShoulder - 50, 450), (0, 255, 0), 3)

				if np.abs(xRShoulder - xRKnee) <= 25:
					calib[2] = 1
					cv2.line(img, (xRHip, yRHip), (xRKnee, yRKnee), (255, 0, 0), 3)
				else:
					calib[2] = 0
					cv2.line(img, (xRHip, yRHip), (xRKnee, yRKnee), (0, 255, 0), 3)

				if np.abs(xLShoulder - xLKnee) <= 25:
					calib[3] = 1
					cv2.line(img, (xLHip, yLHip), (xLKnee, yLKnee), (255, 0, 0), 3)
				else:
					calib[3] = 0
					cv2.line(img, (xLHip, yRHip), (xLKnee, yLKnee), (0, 255, 0), 3)

				# if np.abs(xLShoulder+25 - (xLToe)) <= 25:
				#     cv2.circle(img,(xLToe,yLToe),10,(0,255,0),-1)
				#     calib[1] = 1;
				# elif xLToe < xLShoulder:
				#     try:
				#         img[yLToe-15:yLToe+15,xLToe-15:xLToe+15,:] = arrow1
				#     except:
				#         pass
				# elif xLToe > xLShoulder+25:
				#     try:
				#         img[yLToe-15:yLToe+15,xLToe-15:xLToe+15,:] = arrow2
				#     except:
				#         pass

				if np.sum(calib) == 4:
					calibIdx += 1
					startTime = time.time()
				else:
					calibIdx = 0
					calib[0] = 0
					calib[1] = 0
					disKnee = []

				if calibIdx >= 1:
					cv2.putText(img, "Stay there", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 5)
					disKnee.append(np.abs(xRKnee - xLKnee))

				if calibIdx >= 100:
					status = 1

			if status == 1:
				xRKnee, yRKnee = lmList[26][1:]
				xLKnee, yLKnee = lmList[25][1:]
				xRShoulder, _ = lmList[12][1:]
				xLShoulder, _ = lmList[11][1:]
				xLEar, yLEar = lmList[7][1:]

				currentTime = time.time()

				countFrame += 1
				# if (currentTime - startTime) >= 1 and (currentTime - startTime) <= 2:
				#     cv2.putText(img, "3", (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,0),5)

				# if (currentTime - startTime) > 2 and (currentTime - startTime) <= 3:
				#     cv2.putText(img, "2", (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,0),5)

				# if (currentTime - startTime) > 3 and (currentTime - startTime) <= 4:
				#     cv2.putText(img, "1", (30,80), cv2.FONT_HERSHEY_PLAIN, 7,(0,255,0),5)

				# if (currentTime - startTime) > 4 and (currentTime - startTime) <= 5:
				#     cv2.putText(img, "Go!", (30,80), cv2.FONT_HERSHEY_PLAIN, 7,(0,255,0),5)

				# bar  = np.interp(RightKneeAng,(90,140),(100,450))

				# cv2.rectangle(img,(600,100),(630,450),(0,0,150),3)
				# cv2.rectangle(img,(600,int(bar)),(630,450),(0,0,150),cv2.FILLED)

				# threshKnee = np.mean(disKnee)

				# currentDisKnee = np.abs(xRKnee-xLKnee)

				# if currentDisKnee < threshKnee:
				#     try:
				#         img[yRKnee-15:yRKnee+15,xRKnee-15:xRKnee+15,:] = arrow2
				#     except:
				#         pass
				#     try:
				#         img[yLKnee-15:yLKnee+15,xLKnee-15:xLKnee+15,:] = arrow1
				#     except:
				#         pass

				if RightKneeAng < 100 and xRKnee < xRHip:
					try:
						cv2.putText(img, arrowLR, (80, 120), cv2.FONT_HERSHEY_PLAIN,
									7, (0, 255, 255), 5)
					except:
						pass

				if LeftKneeAng < 100 and xLKnee > xLHip:
					try:
						cv2.putText(img, arrowRL, (80, 120), cv2.FONT_HERSHEY_PLAIN,
									7, (0, 255, 255), 5)
					except:
						pass

				if LeftbackAng < 160:
					try:
						cv2.putText(img, arrowUp, (80, 120), cv2.FONT_HERSHEY_PLAIN,
									7, (0, 255, 255), 5)
					except:
						pass

				# cv2.putText(img, str(LeftbackAng), (30,160), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,255),5)

				# The 75 is the threshold to go down and 170 is for go up
				# Change those number to see which number makes sense to you
				if RightHipAng <= 75 and repDown == 0:
					repDown = 1
					repUp = 0

				if RightHipAng >= 170 and repUp == 0:
					repDown = 0
					repUp = 1
					rep += 1

				cv2.putText(img, str(rep), (30, 80), cv2.FONT_HERSHEY_PLAIN, 7, (0, 255, 255), 5)

		countFrame += 1
		CountAngle = []

		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.


		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		frame_flip = cv2.flip(img,1)
		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes()




