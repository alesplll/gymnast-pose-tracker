from mmdet.apis import init_detector, inference_detector
import numpy as np
import mmcv
import os
import torch


class Detector:
    def __init__(self, config_path=None, checkpoint_path=None, device=None, conf_thresh=0.5):
        config_path = config_path or os.path.join('config', 'cascade_rcnn.py')
        self.device = device or (
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # Всегда используем локальные пути к конфигу и весам
        self.model = init_detector(
            config_path, checkpoint_path, device=self.device)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray):
        """
        frame: np.ndarray (H, W, 3) BGR
        Возвращает: List[{'bbox': [x1, y1, x2, y2], 'score': float}]
        """
        img = frame[..., ::-1]  # BGR->RGB
        results = inference_detector(self.model, img)
        # Универсальная поддержка MMDetection 2.x и 3.x
        if isinstance(results, list):
            # Старый API: results[0] — боксы для класса person
            bboxes = results[0]
            out = []
            for bbox in bboxes:
                if bbox[4] > self.conf_thresh:
                    out.append(
                        {'bbox': bbox[:4].tolist(), 'score': float(bbox[4])})
            return out
        else:
            # Новый API: results — DetDataSample, берем pred_instances напрямую
            pred = getattr(results, 'pred_instances', None)
            if pred is None:
                raise RuntimeError(
                    'MMDetection: не удалось получить pred_instances из результата')
            bboxes = getattr(pred, 'bboxes', None)
            scores = getattr(pred, 'scores', None)
            labels = getattr(pred, 'labels', None)
            if bboxes is None or scores is None or labels is None:
                raise RuntimeError(
                    'MMDetection: не удалось получить bboxes/scores/labels из pred_instances')
            bboxes = bboxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            out = []
            for bbox, score, label in zip(bboxes, scores, labels):
                if label == 0 and score > self.conf_thresh:
                    out.append({'bbox': bbox.tolist(), 'score': float(score)})
            return out
