from collections import deque
import numpy as np
import cv2

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 3 Tennis Ball Tracking')
    parser.add_argument('--video',type=str, help='video file')
    args = parser.parse_args()

    positions = deque()

    if args.video == None:
        feed = cv2.VideoCapture(0)
    else:
        feed  = cv2.VideoCapture(args.video)

    green_lower = np.array([29,50,43],dtype=np.uint8)
    green_upper = np.array([75,255,255],dtype=np.uint8)
    while True:
        ret, frame = feed.read()

        if args.video != None and not ret:
            break

        blurred = cv2.GaussianBlur(frame,(11,11),0)
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv,green_lower,green_upper)
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)

        cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            for c in cnts:
                ((x,y),radius) = cv2.minEnclosingCircle(c)
                # only use large radius objects
                if radius > 15:
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # draw contours
                    cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
                    cv2.circle(frame,center,5,(0,0,255),-1)
                    positions.appendleft(center)


        cv2.imshow("frame",frame)
        cv2.imshow("mask",mask)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

cv2.destroyAllWindows()
