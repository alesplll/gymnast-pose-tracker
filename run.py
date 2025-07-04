from src.pipeline import Pipeline

det_config = 'config/mask_rcnn_r50_fpn_1x_coco.py'
det_ckpt = 'config/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
pose_config = 'config/hrnet_w32_coco_256x192.py'
pose_ckpt = 'config/hrnet_w32_coco_256x192-b9e0b3ab_20200708.pth'

pipeline = Pipeline(det_config, det_ckpt, pose_config,
                    pose_ckpt, device='cpu')  # или 'cpu'

video_path = 'data/0LtLS9wROrk_E_000176_000204.mp4'  # замените на имя вашего файла
output_path = 'output/result.mp4'

pipeline.run(video_path, output_path)
