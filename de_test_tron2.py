import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import argparse, time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image", type=str, help="Path to image to segment")
    p.add_argument("-m", "--model", type=str, help="Model to use", default="COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x.yaml")
    p.add_argument("-t", "--threshold", type=float, help="Threshold for model detections", default=0.4)
    p.add_argument("-rs", "--use_resnest", type=bool, help="Whether the selected model uses ResNeSt backbone or no", default=True)
    return p.parse_args()

def start_segment(args):
    img = args.image
    model = args.model
    thresh = args.threshold
    use_resnest = args.use_resnest

    im = cv2.imread(img)

    # get default cfg file
    cfg = get_cfg()
    # replace cfg from specific model yaml file
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model, resnest=use_resnest)
    predictor = DefaultPredictor(cfg)
    start = time.time()
    outputs = predictor(im)
    print("Time eplased: {}".format(time.time() - start))
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) #rgb image (::-1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    args = parse_args()
    start_segment(args)