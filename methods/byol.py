from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import get_model, get_head
from .base import BaseMethod
from .norm_mse import norm_mse_loss
from typing import Dict, List, Sequence


class BYOL(BaseMethod):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, cfg, wandb):
        """ init additional target and predictor networks """
        super().__init__(cfg)
        self.pred = nn.Sequential(
            nn.Linear(cfg.emb, cfg.head_size),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(),
            nn.Linear(cfg.head_size, cfg.emb),
        )
        self.model_t, _ = get_model(cfg.arch, cfg.dataset)
        self.head_t = get_head(self.out_size, cfg)
        for param in chain(self.model_t.parameters(), self.head_t.parameters()):
            param.requires_grad = False
        self.update_target(0)
        self.byol_tau = cfg.byol_tau
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss

        self.momentum_classfier = nn.Linear(self.out_size, cfg.extra_model_args["numclass"])

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
    
    def _base_forward(self, samples, target, test=False):
        target = target.cuda()
        feats = [self.model(x) for x in samples]
        spmples_length = len(samples)
        with torch.no_grad():
            momentum_feats = [self.model_t(x) for x in samples]

        logits = [self.classfier(x.detach()) for x in feats]
        momentum_logits = [self.momentum_classfier(x.detach()) for x in momentum_feats]

        loss = sum([F.cross_entropy(x, target, ignore_index=-1) for x in logits])
        momentum_loss = sum([F.cross_entropy(x, target, ignore_index=-1) for x in momentum_logits])

        acc = [accuracy_at_k(x, target, top_k=(1, 5)) for x in logits]

        momentum_acc = [accuracy_at_k(x, target, top_k=(1, 5)) for x in momentum_logits]

        log_dict = {"train_acc1":sum([x[0] for x in acc])/spmples_length, 
                    "train_acc5":sum([x[1] for x in acc])/spmples_length,
                    "train_momentum_acc1":sum([x[0] for x in momentum_acc])/spmples_length,
                    "train_momentum_acc5":sum([x[1] for x in momentum_acc])/spmples_length}

        if not test:
            outs = {"feats":feats, "momentum_feats":momentum_feats, 
                    "classfier_loss": loss/spmples_length, "momentum_classfier_loss":momentum_loss/spmples_length}
        else:
            outs = {"classfier_loss": loss/spmples_length, "momentum_classfier_loss":momentum_loss/spmples_length}
            outs.update(log_dict)
            return outs
            
        outs["log_dict"] = log_dict
        return outs


    def forward(self, samples, target):
        outs = self._base_forward(samples, target)
        classfier_loss = outs["classfier_loss"] + outs["momentum_classfier_loss"]
        log_dict = outs["log_dict"]

        feats1, feats2 = outs["feats"]
        momnetum_feats1, momentum_feats2 = outs["momentum_feats"]

        z1 = self.head(feats1)
        z2 = self.head(feats2)

        p1 = self.pred(z1)
        p2 = self.pred(z2)

        with torch.no_grad():
            z1_momentum = self.head_t(momnetum_feats1)
            z2_momentum = self.head_t(momentum_feats2)


        loss = byol_loss_func(p1, z2) + byol_loss_func(p2, z2_momentum) +\
               byol_loss_func(p2, z1) + byol_loss_func(p1, z1_momentum)

        log_dict["byol_loss_step"] = loss
        return loss + classfier_loss, log_dict

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)

        return tau

    def validation_on_m(self, data_loader):
        outs = []
        for n_iter, (samples, target) in enumerate(data_loader):
            # import pdb; pdb.set_trace()

            outs.append(self._base_forward([samples], target, test=True))
        
        val_acc1 = sum([out["train_acc1"] for out in outs])/(n_iter+1)
        val_acc5 = sum([out["train_acc5"] for out in outs])/(n_iter+1)
        val_loss = sum([out["classfier_loss"] for out in outs])/(n_iter+1)

        momentum_val_acc1 = sum([out["train_momentum_acc1"] for out in outs])/(n_iter+1)
        momentum_val_acc5 = sum([out["train_momentum_acc5"] for out in outs])/(n_iter+1)
        momentum_val_loss = sum([out["momentum_classfier_loss"] for out in outs])/(n_iter+1)

        log = {
            "val_acc1":val_acc1,
            "val_acc5":val_acc5,
            "val_loss":val_loss,

            "momentum_val_loss": momentum_val_loss,
            "momentum_val_acc1": momentum_val_acc1,
            "momentum_val_acc5": momentum_val_acc5,
        }

        # import pdb; pdb.set_trace()
        return log


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: BYOL's loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)
