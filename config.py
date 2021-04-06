from yacs.config import CfgNode as CN

cfg = CN()

cfg.SoundEncoder = 'ResNet50'
cfg.SoundReducer = 'SoundMeanPool'

cfg.ImageEncoder = 'ResNet50'
cfg.ImageReducer = 'ImageMeanPool'

cfg.LossFunction = 'TripletLoss'
cfg.LossArg      = []
cfg.LatentDim    = 128

cfg.MarginScaling = False
cfg.LocalizedSampling = False

cfg.DataRoot = './data'

cfg.MaxSamples = 50
cfg.Epochs = 100
cfg.BatchSize = 32
cfg.DataThreads = 4
cfg.AugmentationMode = 'image'

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'FusedAdam'
cfg.Optimizer.LearningRate = 1e-3

cfg.RunId = ''

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0

