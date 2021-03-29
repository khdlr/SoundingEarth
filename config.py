from yacs.config import CfgNode as CN

cfg = CN()

cfg.SoundEncoder = 'ResNet18'
cfg.SoundReducer = 'SoundMeanPool'

cfg.ImageEncoder = 'ResNet18'
cfg.ImageReducer = 'ImageMeanPool'

cfg.LossFunction = 'TripletLoss'
cfg.LossArg      = []
cfg.LatentDim    = 512

cfg.MarginScaling = False
cfg.LocalizedSampling = False

cfg.DataRoot = './data'

cfg.MaxSamples = 50
cfg.Epochs = 100
cfg.BatchSize = 16
cfg.DataThreads = 3
cfg.AugmentationMode = 'image+sound'

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'Adam'
cfg.Optimizer.LearningRate = 1e-3

cfg.RunId = ''

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0

