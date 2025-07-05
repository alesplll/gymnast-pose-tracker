from src.pipeline import Pipeline

pipeline = Pipeline(device='cpu')
pipeline.run('data/temp.mp4', 'output/result.mp4')
