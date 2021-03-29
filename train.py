import sys
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

from einops import rearrange


def full_forward(model, key, img, snd, snd_split, points, metrics):
    img = img.to(dev)
    snd = snd.to(dev)
    points = points.to(dev)

    Z_img = model.img_encoder(img)
    Z_snd = model.snd_encoder(snd, snd_split)

    loss = model.loss_function(Z_img, Z_snd, points)

    with torch.no_grad():
        Z_img = rearrange(Z_img, '(i a) d -> i a d', a=1)
        Z_snd = rearrange(Z_snd, '(i a) d -> i a d', i=1)

        d_matrix = torch.linalg.norm(
            model.loss_function.distance_transform(Z_img) -
            model.loss_function.distance_transform(Z_snd),
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
        torch.save(model.state_dict(), checkpoints / 'latest.pt')


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
    if len(sys.argv) > 1:
        cfg.merge_from_file(sys.argv[1])
    else:
        cfg.merge_from_file("config.yml")
    cfg.freeze()

    img_encoder   = get_model(cfg.ImageEncoder, reducer=cfg.ImageReducer,
        input_dim=3, output_dim=cfg.LatentDim, final_pool=False
    )
    snd_encoder   = get_model(cfg.SoundEncoder, reducer=cfg.SoundReducer,
        input_dim=1, output_dim=cfg.LatentDim, final_pool=True
    )
    loss_function = get_loss_function(cfg.LossFunction)(*cfg.LossArg)
    model = FullModelWrapper(img_encoder, snd_encoder, loss_function)

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    opt = get_optimizer(cfg.Optimizer.Name)(model.parameters(), lr=cfg.Optimizer.LearningRate)

    state.epoch = 0
    state.board_idx = 0
    state.vis_predictions = []

    wandb.init(project='Audiovisual')
    cfg.defrost()
    cfg.RunId = wandb.run.id
    cfg.freeze()
    wandb.config.update(cfg)

    if wandb.run.name:
        log_dir = Path('logs') / wandb.run.name
    else:
        log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()
    with open(log_dir / 'config.yml', 'w') as f:
        print(cfg.dump(), file=f)

    train_data = get_loader(cfg.BatchSize, num_workers=cfg.DataThreads, mode='train')
    val_data = get_loader(cfg.BatchSize, num_workers=cfg.DataThreads, mode='val')

    metrics = Metrics()
    for epoch in range(cfg.Epochs):
        print(f'Starting epoch "{epoch}"')
        train_epoch(model, train_data, metrics)
        val_epoch(model, val_data, metrics)
    evaluate(model, log_dir)

    from downstream.aid import evaluate_aid
    evaluate_aid(model)

    from downstream.aid_few_shot import evaluate_aid_few_shot
    evaluate_aid_few_shot(model)

    from downstream.advance import evaluate_advance
    evaluate_advance(model)

