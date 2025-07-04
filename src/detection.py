import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector


class Detector:
    def __init__(self, config_path, checkpoint_path, device=None, conf_thresh=0.8):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = init_detector(
            config_path, checkpoint_path, device=self.device)
        self.conf_thresh = conf_thresh
        self.person_class_id = 0  # COCO: 0 - person

    def detect(self, frames: np.ndarray):
        """
        frames: np.ndarray (H, W, 3) или (B, H, W, 3)
        Возвращает: List[List[dict]] для batch или List[dict] для одного кадра
        dict: {'bbox': [x1, y1, x2, y2], 'score': float}
        """
        is_batch = frames.ndim == 4
        if not is_batch:
            frames = np.expand_dims(frames, 0)
        results = []
        for frame in frames:
            det_result = inference_detector(self.model, frame)
            bboxes = det_result[0][self.person_class_id]  # только person
            filtered = [
                {'bbox': bbox[:4].tolist(), 'score': float(bbox[4])}
                for bbox in bboxes if bbox[4] > self.conf_thresh
            ]
            results.append(filtered)
        return results if is_batch else results[0]
