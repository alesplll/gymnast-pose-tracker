from src.pipeline import Pipeline

pipeline = Pipeline(device='cpu')
pipeline.run('data/0LtLS9wROrk_E_000176_000204.mp4', 'output/result.mp4')