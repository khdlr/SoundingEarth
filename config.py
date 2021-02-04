from yacs.config import CfgNode as CN

cfg = CN()

cfg.SoundEncoder = 'ResNet18'
cfg.ImageEncoder = 'ResNet18'
cfg.Matcher      = 'MeanPool'
cfg.LossFunction = 'TripletLoss'
cfg.LatentDim    = 128

cfg.MarginScaling = False
cfg.LocalizedSampling = False

cfg.DataRoot = './data'

cfg.MaxSamples = 50
cfg.Epochs = 100
cfg.BatchSize = 32
cfg.DataThreads = 4

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'Adam'
cfg.Optimizer.LearningRate = 1e-3

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0
