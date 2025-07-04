import cv2
import time
import numpy as np
from src.detection import Detector
from src.tracking import Tracker
from src.pose_estimation import PoseEstimator
from src.visualization import draw_bboxes, draw_skeletons


class Pipeline:
    def __init__(self, device=None):
        self.detector = Detector(device=device)
        self.tracker = Tracker()
        self.pose_estimator = PoseEstimator()

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        frame_idx = 0
        t0 = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Детекция + keypoints
            dets = self.detector.detect(frame)
            bboxes = [d['bbox'] for d in dets]
            # Трекинг
            tracked = self.tracker.update(frame, [{'bbox': d['bbox'], 'score': d['score']} for d in dets])
            tracked_bboxes = [t['bbox'] for t in tracked]
            # Сопоставление keypoints с треками (по IoU)
            poses = []
            for t in tracked:
                # ищем ближайший bbox из детекции
                best_det = None
                best_iou = 0
                for d in dets:
                    iou = self._iou(t['bbox'], d['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = d
                if best_det and best_iou > 0.5:
                    poses.append({'keypoints': best_det['keypoints'], 'bbox': t['bbox']})
                else:
                    poses.append({'keypoints': np.zeros((17,3)), 'bbox': t['bbox']})
            # Визуализация
            vis_frame = frame.copy()
            vis_frame = draw_bboxes(vis_frame, tracked_bboxes)
            vis_frame = draw_skeletons(vis_frame, poses)
            out.write(vis_frame)
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}")
        cap.release()
        out.release()
        t1 = time.time()
        print(f"Done. {frame_idx} frames, FPS: {frame_idx/(t1-t0):.2f}")

    @staticmethod
    def _iou(boxA, boxB):
        # [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
