import cv2
import sys

if len(sys.argv) < 2:
    print "Usage: %s [filename]" % sys.argv[0]
    sys.exit(0)

vid = cv2.VideoCapture(sys.argv[1])
print "Press q to exit"
while vid.isOpened():
    _, frame = vid.read()
    cv2.imshow('Load Movie: %s' % sys.argv[1],frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 113:
        break
