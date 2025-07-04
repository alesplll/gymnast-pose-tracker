import numpy as np


def compute_precision_recall(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    gt_boxes, pred_boxes: List[List[x1, y1, x2, y2]]
    Возвращает: precision, recall
    """
    # TODO: реализовать через pycocotools
    return 0.0, 0.0


def compute_mota(gt_tracks, pred_tracks):
    """
    gt_tracks, pred_tracks: списки треков (dict с bbox и track_id)
    Возвращает: MOTA, ID switches
    """
    # TODO: реализовать через motmetrics
    return 0.0, 0


def compute_pck(gt_keypoints, pred_keypoints, thr=0.2):
    """
    gt_keypoints, pred_keypoints: np.ndarray (N, 17, 3)
    Возвращает: PCK@thr
    """
    # TODO: реализовать
    return 0.0


def compute_mae(gt_keypoints, pred_keypoints):
    """
    gt_keypoints, pred_keypoints: np.ndarray (N, 17, 2)
    Возвращает: MAE (mean absolute error) по координатам
    """
    # TODO: реализовать
    return 0.0
