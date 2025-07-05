import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model


class PoseEstimator:
    def __init__(self, config_path, checkpoint_path, device=None):
        self.device = device or ('cpu')
        self.model = init_model(
            config=config_path,
            checkpoint=checkpoint_path,
            device=self.device
        )

    def estimate(self, frame: np.ndarray, bboxes: list):
        """
        frame: np.ndarray (H, W, 3)
        bboxes: List[[x1, y1, x2, y2]]
        Возвращает: List[{'keypoints': np.ndarray (17, 3), 'bbox': [
                                                   x1, y1, x2, y2]}]
        """
        # Фильтруем валидные боксы (не None и с корректной длиной)
        valid = [(i, bbox) for i, bbox in enumerate(bboxes)
                 if bbox is not None and len(bbox) == 4]

        if not valid:
            return []

        indices, valid_bboxes = zip(*valid)
        person_results = [
            {'bbox': np.array(bbox).astype(float)} for bbox in valid_bboxes]

        # Запускаем инференс
        pose_results = inference_topdown(self.model, frame, person_results)

        out = []
        for pose, bbox in zip(pose_results, valid_bboxes):
            # В новых версиях mmpose ключевые точки могут быть в pose['keypoints'] или pose.pred_instances
            # Попробуем получить keypoints из pose['keypoints'], если нет - из pred_instances
            keypoints = None
            if isinstance(pose, dict):
                keypoints = pose.get('keypoints', None)
            else:
                # Если pose - объект с pred_instances
                keypoints = getattr(pose, 'pred_instances', None)
                if keypoints is not None:
                    keypoints = keypoints.get('keypoints', None)

            if keypoints is None:
                # Если ключевые точки не найдены, пропускаем
                continue

            # Если keypoints - тензор, переводим в numpy
            if torch.is_tensor(keypoints):
                keypoints = keypoints.cpu().numpy()

            # Если keypoints имеет лишнее измерение, убираем его
            if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]

            out.append({'keypoints': keypoints, 'bbox': bbox})

        return out
