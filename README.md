# SoundingEarth
Code for [https://github.com/khdlr/SoundingEarth/blob/master/config.yml](https://arxiv.org/abs/2108.00688)

## Data
Download the dataset at https://zenodo.org/record/5600379 and configure `DataRoot` in `config.yml` to point to the extracted archive.

## Code

* `train.py` is the main engine for the project.
* Configuration is done in `config.py` (defaults) and `config.yml` (experiments)
* Data loading in `data_loading.py`

All other interesting stuff goes in `lib/`:
* Models to `lib/models/`
* Loss functions in `lib/loss_functions.py`
