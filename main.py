from Tracking.sort.tracking import Sort
from Detection.yolov5.detect import Detection
import numpy as np
import sys
sys.path.append('Detection')
sys.path.append('Tracking')


def Detect(detector, frame):
    '''
    input : detector, cv2 frame 
    output : numpy boxes (left,top, right,bottom) , numpy scores  
    '''
    box_detects, classes, confs = detector.detect(frame.copy())
    return np.array(box_detects).astype(int), np.array(confs)


def ProcessTracking(video, detector, tracker, skip_frame=1):

    while True:
        _, frame = video.read()
        if(frame is None):
            break

        box_detects, scores = Detect(detector, frame)

        tracked_objects = tracker.update(box_detects)
        for box in tracked_objects:
            print(box)
