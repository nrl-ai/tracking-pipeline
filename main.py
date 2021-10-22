
import cv2
from tools.colors import _COLORS
import numpy as np
import yaml
import sys
sys.path.insert(0, 'Detection')
sys.path.insert(0, 'Tracking')


def VisTracking(img, data_track, labels):
    '''
    input : data_track [[left,top, right,bottom,id_track]]
    output : cv2 show image
    '''

    for i in range(len(data_track)):
        box = data_track[i][:4]
        track_id = int(data_track[i][4])
        cls_id = int(data_track[i][5])

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[track_id % 30] * 255).astype(np.uint8).tolist()
        text = labels[cls_id]+"_"+str(track_id)
        txt_color = (0, 0, 0) if np.mean(
            _COLORS[track_id % 30]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[track_id % 30] * 255 *
                        0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    cv2.imshow("image", img)



def Detect(detector, frame):
    '''
    input : detector, cv2 frame
    output : numpy boxes (left,top, right,bottom) , numpy scores
    '''
    box_detects, classes, confs = detector.detect(frame.copy())
    return np.array(box_detects).astype(int), np.array(confs), np.array(classes)


def ProcessTracking(video, detector, tracker, deep=False, skip_frame=1):
    '''
    output detector.detect : box_detects, classes, confs
            box_detects : [[left,top, right,bottom]]
            classes : [[label1],...]
            confs : [[conf1]...]
    input track : numpy box_detects , numpy confs
    output track : [left,top, right,bottom,track_id,cls]
    '''
    frame_id = 0
    while True:
        _, frame = video.read()
        if(frame is None):
            break
        if(frame_id % skip_frame == 0):

            box_detects, scores, classes = Detect(detector, frame)
            if deep:
                data_track = tracker.update(
                    box_detects, scores, classes, frame.copy())
            else:
                data_track = tracker.update(box_detects, scores, classes)

            VisTracking(frame.copy(), data_track, labels=detector.names)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_id = (frame_id+1) % skip_frame


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


if __name__ == "__main__":
    with open("tracking_config.yaml") as fp:
        config_tracking = yaml.load(fp)
    deep = False

    obj_dt = config_tracking["Object_detection"]["model"]
    obj_tk = config_tracking["Object_tracking"]["model"]

    video = cv2.VideoCapture("videos/palace.mp4")

    if(obj_dt == "yolov5"):
        from Detection.yolov5.detect import Yolov5
        detector = Yolov5(list_objects=["person"])

    elif(obj_dt == "nanodet"):
        from Detection.nanodet.detect import NanoDet
        detector = NanoDet()
    elif(obj_dt == "yolov4"):
        from Detection.yolov4.detect import Yolov4
        detector = Yolov4()
    elif(obj_dt == "yolox"):
        from Detection.yolox.detect import YoloX
        detector = YoloX()



    

    if(obj_tk == "sort"):
        from Tracking.sort.tracking import Sort
        tracker = Sort()

    elif(obj_tk == "norfair"):
        from Tracking.norfair import Norfair
        tracker = Norfair(distance_function=euclidean_distance,
                          distance_threshold=30)

    elif(obj_tk == "motpy"):
        from Tracking.motpy import Motpy
        tracker = Motpy(dt=1/30,
                        model_spec={
                            # position is a center in 2D space; under constant velocity model
                            'order_pos': 1, 'dim_pos': 2,
                            # bounding box is 2 dimensional; under constant velocity model
                            'order_size': 0, 'dim_size': 2,
                            'q_var_pos': 1000.,  # process noise
                            'r_var_pos': 0.1  # measurement noise
                        })

    elif(obj_tk == "bytetrack"):
        from Tracking.bytetrack import BYTETracker

        tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                              match_thresh=0.8, min_box_area=10, frame_rate=30)
    elif(obj_tk == "deepsort"):
        from Tracking.deep_sort import DeepSort
        tracker = DeepSort(model_path="Tracking/deep_sort/deep/checkpoint/ckpt.t7", max_dist=0.2,
                           min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True)
        deep = True

    ProcessTracking(video, detector, tracker, deep)
