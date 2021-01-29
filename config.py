from yacs.config import CfgNode as CN

cfg = CN()

cfg.SoundEncoder = 'ResNet50'
cfg.ImageEncoder = 'ResNet50'
cfg.Matcher      = 'MeanPool'
cfg.LossFunction = 'ContrastiveLoss'
cfg.LatentDim    = 128

cfg.MaxSamples = 50
cfg.Epochs = 300
cfg.BatchSize = 32
cfg.DataThreads = 4

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'Adam'
cfg.Optimizer.LearningRate = 1e-3

cfg.Vis = [0, 14, 17, 37, 49, 89]

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0
