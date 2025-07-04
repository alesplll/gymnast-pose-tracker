import cv2
import numpy as np

# Цвета для скелета (COCO 17 keypoints)
COCO_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # правая рука
    (0, 5), (5, 6), (6, 7), (7, 8),      # левая рука
    (0, 9), (9, 10), (10, 11),           # правая нога
    (0, 12), (12, 13), (13, 14),         # левая нога
    (0, 15), (15, 16)                    # голова
]

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def draw_bboxes(frame, bboxes, color=(0, 255, 0), thickness=2):
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_skeletons(frame, poses, kpt_thr=0.2):
    for pose in poses:
        keypoints = pose['keypoints']  # (17, 3)
        for idx, (i, j) in enumerate(COCO_PAIRS):
            if keypoints[i, 2] > kpt_thr and keypoints[j, 2] > kpt_thr:
                pt1 = tuple(map(int, keypoints[i, :2]))
                pt2 = tuple(map(int, keypoints[j, :2]))
                color = COLORS[idx % len(COLORS)]
                cv2.line(frame, pt1, pt2, color, 2)
        # Нарисовать точки
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > kpt_thr:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
    return frame
