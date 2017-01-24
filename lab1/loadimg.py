import cv2
import sys

if len(sys.argv) < 2:
    print "Usage: %s [filename]" % sys.argv[0]
    sys.exit(0)

img = cv2.imread(sys.argv[1])
cv2.imshow("load Image: %s" % sys.argv[1],img)
print "Press q to exit"
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break
