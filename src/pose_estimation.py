class PoseEstimator:
    def __init__(self, *args, **kwargs):
        pass

    def estimate(self, frame, detections):
        """
        detections: List[{'bbox': ..., 'score': ..., 'keypoints': ...}]
        Возвращает: List[{'keypoints': ..., 'bbox': ...}]
        """
        # Просто возвращаем keypoints из детектора
        return [{'keypoints': d['keypoints'], 'bbox': d['bbox']} for d in detections]
