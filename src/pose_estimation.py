import numpy as np


class PoseEstimator:
    def __init__(self, *args, **kwargs):
        pass

    def estimate(self, frame, detections):
        """
        detections: List[{'bbox': ..., 'score': ..., 'keypoints': ...}]
        Возвращает: List[{'keypoints': ..., 'bbox': ...}]
        """
        # Фильтруем детекции с хорошими keypoints
        filtered_detections = []
        for d in detections:
            keypoints = d['keypoints']
            # Проверяем качество keypoints
            visible_points = np.sum(keypoints[:, 2] > 0.3)
            if visible_points >= 3:  # минимум 3 видимые точки
                filtered_detections.append(d)

        # Просто возвращаем keypoints из детектора
        return [{'keypoints': d['keypoints'], 'bbox': d['bbox']} for d in filtered_detections]
