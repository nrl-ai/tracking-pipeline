import numpy as np
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
import torch
import cv2
import glob
import os
import sys
path_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "Detection/yolov5")


class Yolov5():
    def __init__(self, list_objects=None):  # if

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.path_model = os.path.join(path_cur, "weights/yolov5s.pt")
        self.model = attempt_load(self.path_model, map_location="cuda")
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        if(list_objects is None):
            self.list_objects = self.names
        else:
            self.list_objects = list_objects
        print(self.list_objects)
        print("Detection use ", self.device)
        self.img_size = 640
        self.conf_thres = 0.35
        self.iou_thres = 0.35

    def detect(self, im0s, margin=0, vis=False):
        '''
            input : 
                margin : margin when crop image
                im0s : image input
                vis : draw output and show
            output :
                box_detects : [[left,top,right,bottom],...]
                classes : [label object1 ,label object2...]
                confs : [confidence object1,....]

        '''
        w, h = im0s.shape[:2]
        img = letterbox(im0s.copy(), new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects = []
        classes = []
        confs = []
        cls_ids = []
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=None, agnostic=None)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):
                    if self.names[int(cls)] in self.list_objects:
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        top = max(c1[1]-margin, 0)
                        left = max(c1[0]-margin, 0)
                        right = min(c2[0]+margin, h)
                        bottom = min(c2[1]+margin, w)
                        # box_detects.append(
                        #     [(left+right)//2, (top+bottom)//2, right-left, bottom-top])
                        box_detects.append(
                            [left, top, right, bottom])
                        classes.append(self.names[int(cls)])
                        confs.append([conf.item()])
                        cls_ids.append([int(cls)])
        if(vis):
            img = im0s
            font = cv2.FONT_HERSHEY_SIMPLEX
            for box, lb in zip(box_detects, classes):
                img = cv2.rectangle(img, (box[0], box[1]),
                                    (box[2], box[3]), (0, 255, 0), 3, 1)
                img = cv2.putText(
                    img, lb, (box[0], box[1]), font, 2, (255, 0, 0), 1)
            cv2.imshow("image", img)
            cv2.waitKey(0)

        return box_detects, cls_ids, confs  # bbox_xywh,cls_ids, cls_conf
