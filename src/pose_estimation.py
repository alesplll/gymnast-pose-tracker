import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model


class PoseEstimator:
    def __init__(self, config_path, checkpoint_path, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = init_pose_model(
            config_path, checkpoint_path, device=self.device)

    def estimate(self, frame: np.ndarray, bboxes: list):
        """
        frame: np.ndarray (H, W, 3)
        bboxes: List[[x1, y1, x2, y2]]
        Возвращает: List[{'keypoints': np.ndarray (17, 3), 'bbox': [x1, y1, x2, y2]}]
        """
        person_results = [
            {'bbox': bbox} for bbox in bboxes
        ]
        pose_results, _ = inference_top_down_pose_model(
            self.model, frame, person_results, bbox_thr=None, format='xyxy', return_heatmap=False, outputs=None
        )
        out = []
        for pose, bbox in zip(pose_results, bboxes):
            keypoints = pose['keypoints']  # (17, 3): x, y, score
            out.append({'keypoints': keypoints, 'bbox': bbox})
        return out
