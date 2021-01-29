from yacs.config import CfgNode as CN

cfg = CN()

cfg.Model = 'DeepChanVese'

cfg.Bands = [
    'SPECTRAL/BANDS/NORM_B1_8b',
    'SPECTRAL/BANDS/NORM_B2_8b',
    'SPECTRAL/BANDS/NORM_B3_8b',
    'SPECTRAL/BANDS/NORM_B4_8b',
    'SPECTRAL/BANDS/NORM_B5_8b',
    'SPECTRAL/BANDS/NORM_B6_8b',
    'SPECTRAL/BANDS/NORM_B7_8b',
    'SPECTRAL/BANDS/NORM_B8_8b',
    'SPECTRAL/BANDS/NORM_B10_8b',
    'SPECTRAL/BANDS/NORM_B11_8b',
    # 'TEXTURE/GLCM/11x11_ASM',
    # 'TEXTURE/GLCM/11x11_CON',
    # 'TEXTURE/GLCM/11x11_COR',
    # 'TEXTURE/GLCM/11x11_DIS',
    # 'TEXTURE/GLCM/11x11_ENT',
    # 'TEXTURE/GLCM/11x11_HOM',
    # 'DEM/bed_30m',
    # 'DEM/elevation_ortho_30m',
]

cfg.Loss = 'Hinge'
cfg.Epochs = 30
cfg.BatchSize = 16

cfg.DataThreads = 4

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'Adam'
cfg.Optimizer.LearningRate = 1e-3

cfg.Vis = [0, 300, 710, 900, 1149, 1782]

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0
