import numpy as np
from typing import List, Dict

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    BYTETracker = None  # Требуется установка yolox[tracker] или bytetrack


class Tracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        if BYTETracker is None:
            raise ImportError(
                "BYTETracker не установлен. Установите yolox[tracker] или bytetrack.")
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate
        )

    def update(self, frame: np.ndarray, detections: List[Dict]):
        """
        detections: List[{'bbox': [x1, y1, x2, y2], 'score': float}]
        Возвращает: List[{'track_id': int, 'bbox': [x1, y1, x2, y2], 'score': float}]
        """
        if not detections:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.array([
                det['bbox'] + [det['score']] for det in detections
            ], dtype=np.float32)
        # ByteTrack ожидает dets: (N, 5) — [x1, y1, x2, y2, score]
        online_targets = self.tracker.update(dets, frame)
        results = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            score = t.score
            bbox = [float(tlwh[0]), float(tlwh[1]), float(
                tlwh[0]+tlwh[2]), float(tlwh[1]+tlwh[3])]
            results.append(
                {'track_id': track_id, 'bbox': bbox, 'score': score})
        return results
