

import cv2
import sys
sys.path.insert(0, 'Detection')
from Detection.yolov5.detect import Detection

img = cv2.imread("a.jpeg")
detector = Detection()

box_detects, classes, confs = detector.detect(img, vis=True)
