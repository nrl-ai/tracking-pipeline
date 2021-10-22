
import sys
import os
sys.path.insert(0, "Detection/yolov4")
path_cur = os.path.dirname(os.path.abspath(__file__))
import cv2
from yolov4.tool.torch_utils import *
from yolov4.tool.darknet2pytorch import Darknet
import torch
from yolov4.tool.utils import *


class Yolov4:
    def __init__(self, list_objects=None, cfg="cfg/yolov4-tiny.cfg", path_weight="weights/yolov4-tiny.weights", classes_txt="data/yolo.names"):
        cfg = os.path.join(path_cur, cfg)
        path_weight = os.path.join(path_cur, path_weight)
        classes_txt = os.path.join(path_cur, classes_txt)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(cfg)
        self.model.load_weights(path_weight)
        self.model.to(self.device)
        self.names = load_class_names(classes_txt)
        self.size = (self.model.width, self.model.height)
        self.num_classes = len(self.names)
        if(list_objects is None):
            self.list_objects = self.names
        else:
            self.list_objects = list_objects
        self.conf_thres = 0.35
        self.iou_thres = 0.35

    def detect(self, img):
        img0 = img.copy()
        height,width = img.shape[:2]
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.model, img, self.conf_thres,
                          self.iou_thres, use_cuda=self.device == torch.device("cuda"))[0]
        box_detects = []
        confs = []
        cls_ids = []

        for box in boxes:
            if(self.names[box[6]] in self.list_objects):
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
                box_detects.append([x1, y1, x2, y2])
                cls_ids.append([box[6]])
                confs.append([box[5]])
        # img=plot_boxes_cv2(img0,boxes)

        return box_detects, cls_ids, confs  # bbox_xywh,cls_ids, cls_conf


if __name__ == "__main__":
    X = Yolov4()
    img = cv2.imread("/home/haobk/a.jpeg")
    print(X.detect(img))
