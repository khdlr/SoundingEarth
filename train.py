import torch
import wandb
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from lib import get_optimizer, get_model, get_loss_function, Metrics
from lib.models import FullModelWrapper
from lib.evaluation import evaluate
from config import cfg, state
from data_loading import get_loader


def full_forward(model, key, img, snd, snd_split, distance, metrics):
    img = img.to(dev)
    snd = snd.to(dev)
    # margin = torch.clamp(distance / 100, max=1).to(dev)

    Z_img = model.img_encoder(img)
    Z_snd_multi = model.snd_encoder(snd)

    M_img, M_snd = model.matcher(Z_img, Z_snd_multi, snd_split)

    loss = model.loss_function(M_img, M_snd)

    with torch.no_grad():
        d_matrix = torch.linalg.norm(
            model.loss_function.distance_transform(M_img) -
            model.loss_function.distance_transform(M_snd),
        ord=2, dim=2)

        rk_i2s = 1.0 + d_matrix.argsort(dim=0).argsort(dim=0).diag().float()
        rk_s2i = 1.0 + d_matrix.argsort(dim=1).argsort(dim=1).diag().float()

        N = d_matrix.shape[0]
        d_true = torch.mean(d_matrix.diag())
        d_false = (torch.mean(d_matrix) - d_true / N) * (N / (N-1))

        res = {
            'Loss': loss,
            'I2S: R@1': 100 * (rk_i2s < 1.5).float().mean(),
            'I2S: MedR': rk_i2s.median(),
            'I2S: AvgR': rk_i2s.mean(),
            'S2I: R@1': 100 * (rk_s2i < 1.5).float().mean(),
            'S2I: MedR': rk_s2i.median(),
            'AvgMargin': d_false - d_true,
        }

        metrics.step(**res)
        metrics.step_hist(**{
            'Image2Sound': rk_i2s,
            'Sound2Image': rk_s2i,
        })

    return res


def train_epoch(model, data_loader, metrics):
    state.Epoch += 1
    model.train(True)
    metrics.reset()
    # torch.autograd.set_detect_anomaly(True)
    for iteration, data in enumerate(tqdm(data_loader)):
        res = full_forward(model, *data, metrics)
        opt.zero_grad()
        res['Loss'].backward()
        opt.step()
        state.BoardIdx += data[0].shape[0]

    metrics_vals, metrics_hist = metrics.evaluate()
    logstr = ', '.join(f'{k}: {v:2f}' for k, v in metrics_vals.items())
    print(f'Epoch {state.Epoch:03d} Trn: {metrics_vals}')
    m = {f'trn/{met}': val for met, val in metrics_vals.items()}
    m['_epoch'] = state.Epoch
    for h in metrics_hist:
        m[f'trn/{h}'] = wandb.Histogram(np_histogram=metrics_hist[h])
    wandb.log(m, step=state.BoardIdx)
    wandb.log

    # Save model Checkpoint
    if state.Epoch % 20 == 0:
        torch.save(model.state_dict(), checkpoints / f'{state.Epoch:02d}.pt')


@torch.no_grad()
def val_epoch(model, data_loader, metrics):
    model.train(False)
    metrics.reset()
    for iteration, data in enumerate(data_loader):
        res = full_forward(model, *data, metrics)

    metrics_vals, metrics_hist = metrics.evaluate()
    logstr = ', '.join(f'{k}: {v:2f}' for k, v in metrics_vals.items())
    print(f'Epoch {state.Epoch:03d} Val: {metrics_vals}')
    m = {f'val/{met}': val for met, val in metrics_vals.items()}
    m['_epoch'] = state.Epoch
    for h in metrics_hist:
        m[f'val/{h}'] = wandb.Histogram(np_histogram=metrics_hist[h])
    wandb.log(m, step=state.BoardIdx)


if __name__ == "__main__":
    cfg.merge_from_file("config.yml")
    cfg.freeze()

    img_encoder   = get_model(cfg.ImageEncoder)(
        input_dim=3, output_dim=cfg.LatentDim,
        final_pool=False
    )
    snd_encoder   = get_model(cfg.SoundEncoder)(
        input_dim=1, output_dim=cfg.LatentDim,
        final_pool=True
    )
    matcher       = get_model(cfg.Matcher)()
    loss_function = get_loss_function(cfg.LossFunction)()
    model = FullModelWrapper(img_encoder, snd_encoder, matcher, loss_function)

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    opt = get_optimizer(cfg.Optimizer.Name)(model.parameters(), lr=cfg.Optimizer.LearningRate)

    state.epoch = 0
    state.board_idx = 0
    state.vis_predictions = []

    wandb.init(project='HearingOurPlanet')
    wandb.config.update(cfg)

    if wandb.run.name:
        log_dir = Path('logs') / wandb.run.name
    else:
        log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    train_data = get_loader(cfg.BatchSize, num_workers=cfg.DataThreads, mode='train')
    val_data = get_loader(cfg.BatchSize, num_workers=1, mode='val')

    metrics = Metrics()
    for epoch in range(cfg.Epochs):
        print(f'Starting epoch "{epoch}"')
        train_epoch(model, train_data, metrics)
        val_epoch(model, val_data, metrics)
    evaluate(model, log_dir)
