#!/usr/bin/env python3

from tracking.three_stage_tracker import ThreeStageTracker
import sys
import cv2
import PIL
import numpy as np
import vot_helper


class SiamRCNN:
    def __init__(self, image, region):
        sp = __file__.split("/")
        ckpt = "/".join(sp[:-1]) + "/train_log/hard_mining3/model-1360500"
        self._tracker = ThreeStageTracker(model="checkpoint:" + ckpt)
        x, y, w, h = region
        box = np.array([x, y, w, h])
        self._tracker.init(image, box)

    def track(self, image):
        new_box, score = self._tracker.update(image, use_confidences=True)
        x, y, w, h = new_box
        print(new_box, score)
        rect = vot_helper.Rectangle(x, y, w, h)
        return rect, score


handle = vot_helper.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = np.array(PIL.Image.open(imagefile))
tracker = SiamRCNN(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = np.array(PIL.Image.open(imagefile))
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
