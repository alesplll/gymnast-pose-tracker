import cv2
import time
import numpy as np
from src.detection import Detector
from src.tracking import Tracker
from src.pose_estimation import PoseEstimator
from src.visualization import draw_bboxes, draw_skeletons


class Pipeline:
    def __init__(self, device=None, detector_config=None, detector_weights=None, pose_config=None, pose_weights=None):
        self.detector = Detector(
            config_path=detector_config, checkpoint_path=detector_weights, device=device)
        self.tracker = Tracker()
        self.pose_estimator = PoseEstimator(
            config_path=pose_config, checkpoint_path=pose_weights, device=device)
        self.last_good_poses = []

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
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
            print(f'[F{frame_idx}] Detections: {len(dets)} | {dets[:2]}')
            # Трекинг
            tracked = self.tracker.update(
                frame, [{'bbox': d['bbox'], 'score': d['score']} for d in dets])
            print(f'[F{frame_idx}] Tracked: {len(tracked)} | {tracked[:2]}')
            if not tracked and self.tracker.last_tracks:
                tracked = self.tracker.last_tracks
            # Оценка позы только для треков
            # Передаём все треки как боксы для ViTPose
            tracked_dets = [{'bbox': t['bbox'], 'score': 1.0} for t in tracked]
            print(
                f'[F{frame_idx}] Tracked dets for pose: {len(tracked_dets)} | {tracked_dets[:2]}')
            poses = self.pose_estimator.estimate(frame, tracked_dets)
            print(f'[F{frame_idx}] Poses: {len(poses)} | {poses[:2]}')
            # Фильтрация по качеству keypoints
            good_poses = [p for p in poses if np.sum(
                p['keypoints'][:, 2] > 0.3) >= 1]
            print(
                f'[F{frame_idx}] Good poses: {len(good_poses)} | {good_poses[:2]}')
            if good_poses:
                self.last_good_poses = good_poses
            else:
                good_poses = self.last_good_poses
            # Визуализация
            vis_frame = frame.copy()
            good_bboxes = [p['bbox'] for p in good_poses]
            print(
                f'[F{frame_idx}] Good bboxes: {len(good_bboxes)} | {good_bboxes[:2]}')
            vis_frame = draw_bboxes(vis_frame, good_bboxes)
            vis_frame = draw_skeletons(vis_frame, good_poses)
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
