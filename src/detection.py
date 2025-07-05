import torch
import torchvision
import numpy as np


class Detector:
    def __init__(self, device=None, conf_thresh=0.8):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.conf_thresh = conf_thresh
        self.person_class_id = 1  # COCO: 1 - person (0 - background)

    def detect(self, frame: np.ndarray):
        """
        frame: np.ndarray (H, W, 3) BGR
        Возвращает: List[{'bbox': [x1, y1, x2, y2], 'score': float}]
        """
        img = torch.from_numpy(
            frame[..., ::-1].copy()).float() / 255.0  # BGR->RGB
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)[0]
        results = []
        for i, label in enumerate(outputs['labels']):
            if label.item() == self.person_class_id and outputs['scores'][i] > self.conf_thresh:
                bbox = outputs['boxes'][i].cpu().numpy().tolist()
                score = float(outputs['scores'][i].cpu().numpy())
                results.append({'bbox': bbox, 'score': score})
        return results
