import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, max_age=120, n_init=2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        self.last_tracks = []

    def update(self, frame: np.ndarray, detections: list):
        """
        frame: np.ndarray (H, W, 3) BGR
        detections: List[{'bbox': [x1, y1, x2, y2], 'score': float}]
        Возвращает: List[{'track_id': int, 'bbox': [x1, y1, x2, y2], 'score': float}]
        """
        # Фильтруем детекции по score
        filtered_dets = [d for d in detections if d['score'] > 0.6]

        dets = [
            (d['bbox'], d['score'], None) for d in filtered_dets
        ]
        tracks = self.tracker.update_tracks(dets, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = [float(x) for x in ltrb]
            score = track.det_conf if hasattr(track, 'det_conf') else 1.0
            results.append(
                {'track_id': track_id, 'bbox': bbox, 'score': score})
        if results:
            self.last_tracks = results
        return results
