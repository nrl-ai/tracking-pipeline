#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import time
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess

import sys
path_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "Detection/yolox")


class YoloX(object):
    def __init__(
        self,
        exp_file="exps/example/yolox_voc/yolox_voc_s.py",
        ckpt_file="weights/yolox_s.pth",
        cls_names=COCO_CLASSES,
        decoder=None,
        fp16=False,
        legacy=False,
    ):
        exp_file = os.path.join(path_cur, exp_file)
        ckpt_file = os.path.join(path_cur, ckpt_file)
        exp = get_exp(exp_file, "")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = exp.get_model().to(self.device)
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.cls_conf = 0.35

    def detect(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        box_detects = []
        confs = []
        cls_ids = []
        output = outputs[0]
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        clsids = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(clsids[i])
            score = scores[i]
            if score < self.cls_conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            box_detects.append([x0, y0, x1, y1])
            cls_ids.append(cls_id)
            confs.append(score)
        return box_detects, cls_ids, confs  # bbox_xywh,cls_ids, cls_conf


if __name__ == "__main__":
    detector = YoloX()
