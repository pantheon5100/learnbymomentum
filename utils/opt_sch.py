from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.optim as optim


def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None

def get_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2, momentum=0.9)

    return optimizer