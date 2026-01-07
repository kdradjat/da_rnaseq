import random
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
#import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import autograd
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import accuracy_score

from utils import *


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    features: (N, D) tensor
    labels: (N,) tensor
    """
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)  # (N, N)
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)  # (N, N)

    # Remove self-similarity
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(features.device)
    mask = mask * logits_mask

    logits = similarity_matrix / temperature

    # logsumexp over negatives
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    loss = -mean_log_prob_pos.mean()
    return loss


def build_contr_loss(
        model,
        source_batch,
        target_batch,
        f_criterion,
        device,
        c_weight=1.0
    ):
    x_source, y_source = source_batch
    x_target, y_target = target_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
    
    # get embeddings
    embeddings_source, outputs_source = model.forward_f(x_source)
    embeddings_target, outputs_target = model.forward_f(x_target)
    
    # classifier loss on source
    #loss_cls = F.cross_entropy(outputs_source, y_source)
    loss_cls_source = F.cross_entropy(outputs_source, y_source)
    loss_cls_target = F.cross_entropy(outputs_target, y_target)
    loss_cls = loss_cls_source + loss_cls_target
    
    # pseudo-label target + confidence filtering
    with torch.no_grad():
        _, outputs_target = model.forward_f(x_target)
        preds_target = F.softmax(outputs_target, dim=1)
        pseudo_labels = preds_target.argmax(dim=1)
        confidences = preds_target.max(dim=1).values
        mask = confidences > 0.9
    
    feats_t_conf = embeddings_target[mask]
    pseudo_labels_conf = pseudo_labels[mask]
        
    # combine source + confident target
    #feats_all = torch.cat([embeddings_source, feats_t_conf], dim=0)
    #labels_all = torch.cat([y_source, pseudo_labels_conf], dim=0)
    feats_all = torch.cat([embeddings_source, embeddings_target], dim=0)
    labels_all = torch.cat([y_source, y_target], dim=0)
    
    # contrastive loss
    loss_contr = supervised_contrastive_loss(feats_all, labels_all)
    
    # total loss
    total_loss = loss_cls + c_weight * loss_contr

    return total_loss, loss_cls, loss_contr


def da_train_epoch_contr(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        f_optimizer,
        device,
        c_weight=1.0
    ):
    model.train()
    train_steps = 1.0
    
    total_loss_epoch, cls_loss_epoch, contr_loss_epoch = 0.0, 0.0, 0.0
    total_loss_steps, cls_loss_steps, contr_loss_steps = [], [], []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        total_loss, cls_loss, contr_loss = build_contr_loss(model, source_batch, target_batch, f_criterion=f_criterion, device=device, c_weight=c_weight)
        f_optimizer.zero_grad()
        total_loss.backward()
        f_optimizer.step()
        
        # logs
        total_loss_epoch += total_loss.item()
        cls_loss_epoch += cls_loss.item()
        contr_loss_epoch += contr_loss.item()
        total_loss_steps.append(total_loss.item())
        cls_loss_steps.append(cls_loss.item())
        contr_loss_steps.append(contr_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps,
            cls_loss_epoch/train_steps,
            contr_loss_epoch/train_steps,
            total_loss_steps,
            cls_loss_steps,
            contr_loss_steps
           ]
    
    