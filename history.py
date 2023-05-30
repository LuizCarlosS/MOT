import numpy as np

from dataset import FrameResults
from tracker import Assigner

class IdHistory:
    def __init__(self, detections: np.ndarray, ids: np.ndarray):
        self.history = []

    def update(self, detections: np.ndarray, assign:Assigner):

        # 1.Match detections with previous history
        macthing_scores = self.match(detections)
        # 2.Assign IDs - If matched, same ID. If unmatched, new ID
        ids = assign(self, macthing_scores)
        # 3.Create FrameResult object.
        frame_res = FrameResults(detections, ids)
        # 4.Add object to history
        pass

    def add(self, frameResult: FrameResults):
        self.history.append(frameResult)

    def match(self, detections: np.ndarray):
        pass