from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from dataset import FrameResults
from yolov7.object_detector import  ObjectDetector

class Tracker(ABC): # This can be some different options
    def __init__(self, model_path:str):
        self.detector = ObjectDetector(model_path)
        self.assigner = Assigner()

    def __call__(self, *args, **kwargs):
        pass

class Assigner(ABC): # This can be some different options
    def __init__(self):
        pass
    def __call__(self, history, detections):
        pass # deals with the detections and history