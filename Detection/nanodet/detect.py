
import os
import sys
import cv2
import torch
sys.path.insert(0, "Detection/nanodet")
path_cur = os.path.dirname(os.path.abspath(__file__))
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_model_weight, load_config



class NanoDet(object):
    def __init__(self, list_objects=None, cfg_file="config/nanodet-m.yml", model_path="weights/nanodet_m.ckpt"):
        model_path = os.path.join(path_cur, model_path)
        cfg_file = os.path.join(path_cur, cfg_file)
        load_config(cfg, cfg_file)
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(cfg.model)
        ckpt = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.score_thresh = 0.35
        self.names = cfg.class_names
        if(list_objects is None):
            list_objects = self.names
        self.pipeline = Pipeline(
            cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def detect(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(
            meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            dets = self.model.inference(meta)[0]

        box_detects = []
        cls_ids = []
        confs = []
        for label in dets:
            for bbox in dets[label]:
                score = bbox[-1]
                if score > self.score_thresh and self.names[label] in self.list_objects:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    box_detects.append([x0, y0, x1, y1])
                    cls_ids.append([label])
                    confs.append([score])

        return box_detects, cls_ids, confs


if __name__ == "__main__":
    detector = NanoDet()
    img = cv2.imread("/home/haobk/a.jpeg")
    box_detects, cls_ids, confs = detector.detect(img)
    print(box_detects, cls_ids, confs)
