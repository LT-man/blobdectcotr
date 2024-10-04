import cv2
import numpy
signs = cv2.imread("signs.jpg")
gray = cv2.cvtColor(signs, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0,0)
cv2.imshow("screen", blur)
cv2.waitKey(0)

parameters = cv2.SimpleBlobDetector_Params()
parameters.filterByArea = True
parameters.filterByCircularity = True
parameters.filterByConvexity = True
parameters.filterByInertia = True
parameters.minArea = 100
parameters.minCircularity = 0.01
parameters.minConvexity = 0.01
parameters.minInertiaRatio = 0.01
model = cv2.SimpleBlobDetector_create(parameters)
points = model.detect(signs)
print(points)
blobs = cv2.drawKeypoints(signs, points, numpy.zeros((1,1)),(49, 132, 223),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("screen3", blobs)
cv2.waitKey(0)