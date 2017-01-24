import cv2
import sys

print "Press q to exit"

cap = cv2.VideoCapture(0)

while(1):
	_, frame = cap.read()

	cv2.imshow('cam',frame)
	k = cv2.waitKey(5) & 0xFF
	if k == 113:
		break
