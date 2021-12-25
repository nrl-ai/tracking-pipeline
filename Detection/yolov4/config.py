from easydict import EasyDict as edict
import torch
import os

def get_config():
    conf=edict()
    conf.name="Yolov4"
    conf.cur_dir=os.path.dirname(os.path.realpath(__file__))
    conf.cfg=os.path.join(conf.cur_dir,'cfg',conf.name+'.cfg')
    conf.names=os.path.join(conf.cur_dir,'data',conf.name+'.names')
    conf.weights=os.path.join(conf.cur_dir,'weights',conf.name+'.weights')
    conf.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return conf
