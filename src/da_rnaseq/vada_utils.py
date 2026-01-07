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

def build_vat_loss(model, x, xi=10.0, eps=1.0, ip=1):
    # Virtual Adversarial Training loss
    with torch.no_grad():
        _, c_outputs = model.forward_f(x)
        #pred = nn.Softmax()(source_outputs)
        pred = F.softmax(c_outputs, dim=1)

    # Initialize random unit noise
    d = torch.randn_like(x)
    d = F.normalize(d, p=2, dim=1)

    for _ in range(ip):
        d.requires_grad_()
        _, pred_hat = model.forward_f(x + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction='batchmean')
        grad = torch.autograd.grad(loss, [d])[0]
        d = F.normalize(grad, p=2, dim=1).detach()

    r_adv = eps * d
    _, pred_hat = model.forward_f(x + r_adv)
    logp_hat = F.log_softmax(pred_hat, dim=1)
    return F.kl_div(logp_hat, pred, reduction='batchmean')

def build_f_loss_vada(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=False, lambda_ent=0.01, lambda_vat=0.1):
    x_source, y_source = source_batch
    x_target, y_target = target_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
    #print(y_source)
    #print(y_target)
    
    if supervised:
        # forward source/target + classification loss
        source_embeddings, source_outputs = model.forward_f(x_source)
        target_embeddings, target_outputs = model.forward_f(x_target)
        #cls_loss = f_criterion(nn.Softmax()(source_outputs), y_source)
        cls_loss_source = f_criterion(source_outputs, y_source)
        cls_loss_target = f_criterion(target_outputs, y_target)
        cls_loss = cls_loss_source + cls_loss_target
        
    else:
        # forward source + classification loss
        source_embeddings, source_outputs = model.forward_f(x_source)
        #cls_loss = f_criterion(nn.Softmax()(source_outputs), y_source)
        cls_loss = f_criterion(source_outputs, y_source)
    
    if supervised:
        ent_loss = 0.0 
    else:
        # forward target + entropy check
        target_embeddings, target_outputs = model.forward_f(x_target)
        p_t = F.softmax(target_outputs, dim=1)
        ent_loss = -torch.mean(torch.sum(p_t * torch.log(p_t + 1e-8), dim=1))
    
    # VAT loss
    vat_loss_source = build_vat_loss(model, x_source)
    vat_loss_target = build_vat_loss(model, x_target)
    vat_loss = vat_loss_source + vat_loss_target
    
    # adversarial loss
    z = torch.cat([source_embeddings, target_embeddings], 0)
    d_z = model.forward_d(z)
    source_dz = d_z[:source_embeddings.shape[0]]
    target_dz = d_z[source_embeddings.shape[0]:]
    fd_source_loss, fd_target_loss = d_criterion(source_dz, target_dz)
    #fd_loss = - fd_source_loss
    adv_loss = - fd_target_loss
    
    total_f_loss = cls_loss + adv_loss + lambda_ent * ent_loss + lambda_vat * vat_loss
    
    return total_f_loss, cls_loss, ent_loss, vat_loss, adv_loss


def build_f_loss_vada_ss(model, source_batch, target_labeled_batch, target_unlabeled_batch, f_criterion, d_criterion, device, lambda_ent=0.01, lambda_vat=0.1):
    x_source, y_source = source_batch
    x_target_1, y_target_1 = target_labeled_batch
    x_target_2, y_target_2 = target_unlabeled_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target_1 = x_target_1.to(device)
    y_target_1 = y_target_1.to(device)
    x_target_2 = x_target_2.to(device)
    y_target_2 = y_target_2.to(device)
    #print(y_source)
    #print(y_target)
    
    
    # forward source/target labeled + classification loss
    source_embeddings, source_outputs = model.forward_f(x_source)
    target_embeddings_1, target_outputs_1 = model.forward_f(x_target_1)
    #cls_loss = f_criterion(nn.Softmax()(source_outputs), y_source)
    cls_loss_source = f_criterion(source_outputs, y_source)
    cls_loss_target = f_criterion(target_outputs_1, y_target_1)
    cls_loss = cls_loss_source + cls_loss_target
        
    # forward target unlabeled + entropy check
    target_embeddings_2, target_outputs_2 = model.forward_f(x_target_2)
    p_t = F.softmax(target_outputs_2, dim=1)
    ent_loss = -torch.mean(torch.sum(p_t * torch.log(p_t + 1e-8), dim=1))
    
    # VAT loss
    vat_loss_source = build_vat_loss(model, x_source)
    vat_loss_target_1 = build_vat_loss(model, x_target_1)
    vat_loss_target_2 = build_vat_loss(model, x_target_2)
    vat_loss = vat_loss_source + vat_loss_target_1 + vat_loss_target_2
    
    # adversarial loss
    target_embeddings = torch.cat([target_embeddings_1, target_embeddings_2], 0)
    z = torch.cat([source_embeddings, target_embeddings], 0)
    d_z = model.forward_d(z)
    source_dz = d_z[:source_embeddings.shape[0]]
    target_dz = d_z[source_embeddings.shape[0]:]
    fd_source_loss, fd_target_loss = d_criterion(source_dz, target_dz)
    #fd_loss = - fd_source_loss
    adv_loss = - fd_target_loss
    
    total_f_loss = cls_loss + adv_loss + lambda_ent * ent_loss + lambda_vat * vat_loss
    
    return total_f_loss, cls_loss, ent_loss, vat_loss, adv_loss



def da_train_epoch_vada(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        supervised,
        n_critic=1
    ):
    
    model.train()
    train_steps = 1.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    total_d_loss_epoch, d_loss_epoch, d_source_loss_epoch, d_target_loss_epoch, d_grad_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    total_d_loss_steps, d_loss_steps, d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps = [], [], [], [], []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # train D
        total_d_loss, d_loss, d_source_loss, d_target_loss, d_grad_loss = build_d_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, double_encoder=False)
        d_optimizer.zero_grad()
        total_d_loss.backward()
        d_optimizer.step()
        
        # D logs
        total_d_loss_epoch += total_d_loss.item()
        d_loss_epoch += d_loss.item()
        d_source_loss_epoch += d_source_loss.item()
        d_target_loss_epoch += d_target_loss.item()
        d_grad_loss_epoch += d_grad_loss.item()
        total_d_loss_steps.append(total_d_loss.item())
        d_loss_steps.append(d_loss.item())
        d_source_loss_steps.append(d_source_loss.item())
        d_target_loss_steps.append(d_target_loss.item())
        d_grad_loss_steps.append(d_grad_loss.item())
        
        # train E + C
        if train_steps % n_critic == 0:
            total_f_loss, cls_loss, ent_loss, vat_loss, adv_loss = build_f_loss_vada(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            f_loss_epoch += cls_loss.item()
            fd_loss_epoch += adv_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
            f_loss_steps.append(cls_loss.item())
            fd_loss_steps.append(adv_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]

# semi-supervised
def da_train_epoch_vada_ss(
        model, 
        source_loader, 
        target_labeled_loader, 
        target_unlabeled_loader,
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        n_critic=1
    ):
    
    model.train()
    train_steps = 1.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    total_d_loss_epoch, d_loss_epoch, d_source_loss_epoch, d_target_loss_epoch, d_grad_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    total_d_loss_steps, d_loss_steps, d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps = [], [], [], [], []
    
    for source_batch, target_labeled_batch, target_unlabeled_batch in zip(source_loader, target_labeled_loader, target_unlabeled_loader):
        # train D
        #target_batch = torch.cat([target_labeled_batch, target_unlabeled_batch], 0)
        target_batch = [torch.cat([target_labeled_batch[0], target_unlabeled_batch[0]], 0), torch.cat([target_labeled_batch[1], target_unlabeled_batch[1]], 0)]
        print(target_batch)
        total_d_loss, d_loss, d_source_loss, d_target_loss, d_grad_loss = build_d_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, double_encoder=False)
        d_optimizer.zero_grad()
        total_d_loss.backward()
        d_optimizer.step()
        
        # D logs
        total_d_loss_epoch += total_d_loss.item()
        d_loss_epoch += d_loss.item()
        d_source_loss_epoch += d_source_loss.item()
        d_target_loss_epoch += d_target_loss.item()
        d_grad_loss_epoch += d_grad_loss.item()
        total_d_loss_steps.append(total_d_loss.item())
        d_loss_steps.append(d_loss.item())
        d_source_loss_steps.append(d_source_loss.item())
        d_target_loss_steps.append(d_target_loss.item())
        d_grad_loss_steps.append(d_grad_loss.item())
        
        # train E + C
        if train_steps % n_critic == 0:
            total_f_loss, cls_loss, ent_loss, vat_loss, adv_loss = build_f_loss_vada_ss(model, source_batch, target_labeled_batch, target_unlabeled_batch, f_criterion, d_criterion, device)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            f_loss_epoch += cls_loss.item()
            fd_loss_epoch += adv_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
            f_loss_steps.append(cls_loss.item())
            fd_loss_steps.append(adv_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]


def build_loss_dirtt(
        model,
        teacher,
        target_batch,
        device,
        beta=1.0,
        lambda_ent=0.01
    ):
    x_target, _ = target_batch
    x_target = x_target.to(device)
    
    _, outputs = model.forward_f(x_target)
    
    with torch.no_grad():
        _, outputs_teacher = teacher.forward_f(x_target)
    
    # entropy
    ent_loss = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
    
    # KL divergence
    kl_div = F.kl_div(F.log_softmax(outputs, dim=1), F.log_softmax(outputs_teacher, dim=1), reduction='batchmean')
    
    loss = lambda_ent * ent_loss + beta * kl_div
    
    return loss

def da_train_epoch_dirtt(
        model,
        teacher,
        optimizer,
        target_loader,
        device,
        beta=1.0,
        lambda_ent=0.01
    ):
    model.train()
    teacher.eval()
    
    train_steps = 1.0
    
    total_loss_epoch = 0.0
    total_loss_steps = []
    
    for target_batch in target_loader:
        dirtt_loss = build_loss_dirtt(model, teacher, target_batch, device=device, beta=beta, lambda_ent=lambda_ent)
        optimizer.zero_grad()
        dirtt_loss.backward()
        optimizer.step()
        
        # logs
        total_loss_epoch += dirtt_loss.item()
        total_loss_steps.append(dirtt_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps, total_loss_steps]