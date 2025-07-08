from mmpose.apis import init_model, inference_topdown
import numpy as np
import os
import torch


class PoseEstimator:
    def __init__(self, config_path=None, checkpoint_path=None, device=None):
        config_path = config_path or os.path.join('config', 'vitpose.py')
        checkpoint_path = checkpoint_path or os.path.join(
            'config', 'vitpose.pth')
        self.device = device or (
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = init_model(
            config_path, checkpoint_path, device=self.device)

    def estimate(self, frame, detections):
        """
        detections: List[{'bbox': ..., 'score': ...}]
        Возвращает: List[{'keypoints': np.ndarray, 'bbox': ...}]
        """
        person_results = [{'bbox': d['bbox']} for d in detections]
        pose_results = inference_topdown(
            self.model, frame[..., ::-1], person_results)
        out = []
        for pose, det in zip(pose_results, detections):
            keypoints = pose.pred_instances.get('keypoints').cpu().numpy()[0]
            out.append({'keypoints': keypoints, 'bbox': det['bbox']})
        return out
