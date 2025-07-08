from src.pipeline import Pipeline

# Пути к локальным конфига и весам
DETECTOR_CONFIG = '/home/wexel/Data/Code/Python/ML/gymnast-pose-tracker/config/CascadeRCNN/cascade-rcnn_r50_fpn_1x_coco.py'
DETECTOR_WEIGHTS = '/home/wexel/Data/Code/Python/ML/gymnast-pose-tracker/config/CascadeRCNN/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
POSE_CONFIG = '/home/wexel/Data/Code/Python/ML/gymnast-pose-tracker/config/ViTPose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
POSE_WEIGHTS = '/home/wexel/Data/Code/Python/ML/gymnast-pose-tracker/config/ViTPose/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'

pipeline = Pipeline(
    device='cuda',
    detector_config=DETECTOR_CONFIG,
    detector_weights=DETECTOR_WEIGHTS,
    pose_config=POSE_CONFIG,
    pose_weights=POSE_WEIGHTS
)
pipeline.run('data/temp.mp4', 'output/result.mp4')
