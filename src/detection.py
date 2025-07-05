import torch
import torchvision
import numpy as np


class Detector:
    def __init__(self, device=None, conf_thresh=0.8):
        self.device = device or ("cpu")
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray):
        """
        frame: np.ndarray (H, W, 3) BGR
        Возвращает: List[{'bbox': [x1, y1, x2, y2], 'score': float, 'keypoints': np.ndarray (17, 3)}]
        """
        img = torch.from_numpy(
            frame[..., ::-1].copy()).float() / 255.0  # BGR->RGB
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)[0]
        results = []
        for i, label in enumerate(outputs['labels']):
            # 1 = person
            if label.item() == 1 and outputs['scores'][i] > self.conf_thresh:
                bbox = outputs['boxes'][i].cpu().numpy().tolist()
                score = float(outputs['scores'][i].cpu().numpy())
                keypoints = outputs['keypoints'][i].cpu().numpy()  # (17, 3)

                # Фильтрация по размеру бокса (минимальный размер 50x50)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 50:
                    continue

                # Проверка качества keypoints (должно быть достаточно видимых точек)
                visible_keypoints = np.sum(
                    keypoints[:, 2] > 0.5)  # confidence > 0.5
                if visible_keypoints < 5:  # минимум 5 видимых точек
                    continue

                results.append(
                    {'bbox': bbox, 'score': score, 'keypoints': keypoints})
        return results
