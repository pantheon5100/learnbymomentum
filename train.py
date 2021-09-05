from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.backends.cudnn as cudnn

from utils.cfg import get_cfg
from datasets import get_ds
from methods import get_method
from utils.version import get_version
from utils.backup_code import backup
from utils.opt_sch import get_scheduler, get_optimizer

import os

if __name__ == "__main__":
    cfg = get_cfg()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    version = get_version()
    cfg.versions["version"] = version
    backup(cfg)

    if cfg.wandb:
        wandb.init(project=cfg.project, config=cfg, entity="kaistssl", name=cfg.name)
    else:
        wandb = None

    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers, dataset_dir=cfg.data_dir)
    model = get_method(cfg.method)(cfg, wandb)
    model.cuda().train()
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    train_loader = ds.train
    clf_loader = ds.clf
    test_loader = ds.test

    global_step = 1
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        iters = len(train_loader)

        for n_iter, (samples, target) in enumerate(tqdm(train_loader, position=1)):
            global_step += 1
            model.train()
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                current_learning_rate = cfg.lr * lr_scale

                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            loss, log_dict = model(samples, target)

            loss.backward()
            optimizer.step()

            loss_ep.append(loss.item())
            tau = model.step(ep / cfg.epoch)
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)
                current_learning_rate = scheduler.get_last_lr()[0]

            if cfg.wandb and global_step % 50 == 0:
                log_dict["momentum_tau"] = tau

                wandb.log(log_dict, step=global_step, commit=False)

        validation_log = model.validation_on_m(test_loader)
        if cfg.wandb:
            wandb.log(validation_log, step=global_step)

        if cfg.lr_step == "step":
            scheduler.step()

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if (ep + 1) % eval_every == 0:
            acc_knn, acc = model.get_acc(clf_loader, test_loader)
            if cfg.wandb:
                wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, step=global_step, commit=False)

        if (ep + 1) % 100 == 0:
            fname = f"data/{cfg.method}_{cfg.dataset}_{ep}.pt"
            torch.save(model.state_dict(), fname)

        if cfg.wandb:
            wandb.log({"loss": np.mean(loss_ep), "lr":current_learning_rate, "ep": ep}, step=global_step)

        # print(f"EPOCH [{ep}/{cfg.epoch}]: acc_knn {acc_knn:2.2%}, acc_linear {acc[1]:2.2%}, loss {loss:2.4%}")


