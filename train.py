import torch
import torch.nn.functional as F
import wandb
from itertools import chain
from pathlib import Path
from datetime import datetime
import shutil
from tqdm import tqdm

from lib import get_optimizer, get_model, get_loss_function, Metrics, plotting
from config import cfg, state
from data_loading import get_loader


def model_forward(model, imagery, ground_truth, metrics):
    imagery = imagery.to(model.device)
    ground_truth = ground_truth.to(model.device)
    prediction = model(imagery)

    loss = loss_fn(prediction, target)

    with torch.no_grad():
        metrics.step(prediction, target)

    return dict(
        images=images,
        ground_truth=ground_truth,
        prediction=prediction,
        loss=loss,
        accuracy=accuracy,
    )


def train_epoch(models, data_loader):
    state.Epoch += 1
    for model in models:
        model.train(True)
    progress = tqdm(data_loader)

    # torch.autograd.set_detect_anomaly(True)
    for iteration, data in enumerate(progress):
        res = full_forward(models, data)
        opt.zero_grad()
        res['loss'].backward()
        opt.step()

        with torch.no_grad():
            metrics.step(**{k: v for k, v in res.items() if 'loss' in k or 'l2' in k})
            if (iteration+1) % 500 == 0:
                metrics_vals = metrics.evaluate()
                wandb.log({f'trn/{met}': val for met, val in metrics_vals.items()}, step=state.BoardIdx)
                state.BoardIdx += 1
                progress.set_postfix_str(f"{metrics_vals['loss']:.2f}")

    # Save model Checkpoint
    for model, name in zip(models, ['encoder', 'decoder', 'transcoder']):
        torch.save(model.state_dict(), checkpoints / f'{epoch:02d}_{name}.pt')


def val_epoch(models, data_loader):
    metrics = Metrics()
    for model in models:
        model.train(False)

    base_idx = 0
    for iteration, data in enumerate(data_loader):
        res = full_forward(models, data)
        metrics.step(**{k: v for k, v in res.items() if 'loss' in k or 'l2' in k})

        images, timestamps, masks, splits = data
        # Image logging
        for i in range(len(splits)):
            if base_idx + i in cfg.Vis:
                image     = torch.split(images, splits)[i]
                timestamp = torch.split(timestamps, splits)[i]
                mask      = torch.split(masks, splits)[i]

                unknown_split = [torch.sum(1 - t.to(torch.int)) for t in torch.split(masks, splits)]
                reconst   = torch.split(res['e2e_reconstruction'], unknown_split)[i]
                log_image(image, timestamp, mask, reconst, tag=f'{base_idx+i}')
        base_idx += len(splits)

    metrics_vals = metrics.evaluate()
    wandb.log({f'val/{met}': val for met, val in metrics_vals.items()}, step=state.BoardIdx)


if __name__ == "__main__":
    cfg.merge_from_file("config.yml")
    cfg.freeze()

    model = get_model(cfg.Model)()
    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    opt = get_optimizer(cfg.Optimizer.Name)(model.parameters(), lr=cfg.Optimizer.LearningRate)

    state.epoch = 0
    state.board_idx = 0
    state.vis_predictions = []

    wandb.init(project='Chan Vese')
    wandb.config.update(cfg)

    if wandb.run.name:
        log_dir = Path('logs') / wandb.run.name
    else:
        log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy('config.yml', log_dir / 'config.yml')
    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    train_data = get_loader(cfg.BatchSize, cfg.DataThreads, mode='train')
    val_data = get_loader(cfg.BatchSize, 1, mode='val')

    metrics = Metrics()
    loss_function = get_loss_function(cfg.Loss)

    for epoch in range(cfg.Epochs):
        print(f'Starting epoch "{epoch}"')
        train_epoch(model, train_data)
        val_epoch(model, val_data)
