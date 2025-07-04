import cv2
import time
import numpy as np
from src.detection import Detector
from src.tracking import Tracker
from src.pose_estimation import PoseEstimator
from src.visualization import draw_bboxes, draw_skeletons


class Pipeline:
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, device=None):
        self.detector = Detector(det_config, det_ckpt, device=device)
        self.tracker = Tracker()
        self.pose_estimator = PoseEstimator(
            pose_config, pose_ckpt, device=device)

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
            # Детекция
            dets = self.detector.detect(frame)
            bboxes = [d['bbox'] for d in dets]
            # Трекинг
            tracked = self.tracker.update(frame, dets)
            tracked_bboxes = [t['bbox'] for t in tracked]
            # Оценка позы
            poses = self.pose_estimator.estimate(frame, tracked_bboxes)
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
