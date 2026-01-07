import random
import numpy as np
import torch
from torch import autograd
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
#import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import autograd
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_epoch(model, criterion, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    train_steps = 0.0

    for batch in train_loader:
        x, y = batch
        x=x.to(device)
        y=y.to(device)
        # get embeddings
        embeddings, outputs = model(x)
        # compute loss
        loss = criterion(outputs, y)
        #loss = criterion(nn.Softmax()(outputs), y)
        loss.backward()
        # update model weights
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()
        # log progress
        epoch_loss += loss.item()
        train_steps += 1

    return epoch_loss / train_steps

def valid_loss_f(model, criterion, train_loader, device) :
    model.eval()
    epoch_loss = 0.0
    valid_steps = 0.0
    
    y_true, y_pred = [], []
    
    for batch in train_loader : 
        x, y = batch
        x=x.to(device)
        y=y.to(device)
        embeddings, outputs = model(x)
        loss = criterion(outputs, y)
        #loss = criterion(nn.Softmax()(outputs), y)
        epoch_loss += loss.item()
        valid_steps += 1
    
    return epoch_loss / valid_steps


def epoch_acc(model, loader, device):
    model.eval()
    epoch_acc = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            embeddings, outputs = model(x)
            
            predictions = nn.Softmax()(outputs).detach().cpu().numpy().argmax(1)
            y_pred += predictions.tolist()
            y_true += y.detach().cpu().numpy().tolist()
    
    epoch_acc = accuracy_score(y_true, y_pred)
    
    return epoch_acc


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (str): 'min' for minimizing loss (default), 'max' for maximizing accuracy, etc.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if self.best_score is None:
            self.best_score = value
        else:
            if self.mode == 'min':
                improvement = self.best_score - value
            else:  # mode == 'max'
                improvement = value - self.best_score

            if improvement > self.min_delta:
                self.best_score = value
                self.counter = 0  # Reset counter on improvement
            else:
                self.counter += 1  # Increase counter if no improvement

            if self.counter >= self.patience:
                self.early_stop = True  # Stop training

        return self.early_stop


#### Domain Adaptation

def soft_relu(x):
    return ((-x.abs()).exp() + 1.0).log() + torch.clip(x, min=0.0)

def wasserstein_beta(d1, d2, beta=0.0):
    """Relaxed Wasserstein distance given dual function outputs.

    minimizing wasserstein_beta gives p2/p1 <= 1 + beta.
    Return:
        d2 - (1 + beta) * d1 where d1, d2 >= 0
    """
    part1 = - (1.0 + beta) * soft_relu(d1).mean()
    part2 = soft_relu(d2).mean()
    return part1 + part2


def da_train_epoch(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device
    ):
    
    model.train()
    train_steps = 0.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    d_loss_epoch = 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    d_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        x_source, y_source = source_batch
        x_target, _ = target_batch
        #print(x_source)
        #print(y_source)
        #print(x_target)
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        
        ### Forward f
        source_embeddings, source_outputs = model.forward_f(x_source)
        f_loss = f_criterion(nn.Softmax()(source_outputs), y_source)
        
        target_embeddings, _ = model.forward_f(x_target)
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        fd_loss = d_criterion(source_dz, target_dz)
        
        total_f_loss = f_loss + d_criterion_w * fd_loss
        
        # update f 
        f_optimizer.zero_grad()
        total_f_loss.backward()
        f_optimizer.step()
        
        # log progress
        total_f_loss_epoch += total_f_loss.item()
        f_loss_epoch += f_loss.item()
        fd_loss_epoch += fd_loss.item()
        total_f_loss_steps.append(total_f_loss.item())
        f_loss_steps.append(f_loss.item())
        fd_loss_steps.append(fd_loss.item())
        
        #total_f_loss = total_f_loss.detach()
        
        
        ### Forward d
        x_source, _ = source_batch
        x_target, _ = target_batch
        x_source = x_source.detach()
        x_target = x_target.detach()
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        with torch.no_grad():
            source_embeddings, _ = model.forward_f(x_source)
            target_embeddings, _ = model.forward_f(x_target)
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        d_loss = - d_criterion(source_dz, target_dz)
        
        # update d
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # log progress
        d_loss_epoch += d_loss.item()
        d_loss_steps.append(d_loss.item())
        
        train_steps += 1
        
    return total_f_loss_epoch/train_steps, f_loss_epoch/train_steps, fd_loss_epoch/train_steps, d_loss_epoch/train_steps, total_f_loss_steps, f_loss_steps, fd_loss_steps, d_loss_steps


    
    
def da_train_epoch_fully_supervised(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device
    ):
    
    model.train()
    train_steps = 0.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    d_loss_epoch = 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    d_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        x_source, y_source = source_batch
        x_target, y_target = target_batch
        #print(x_source)
        #print(y_source)
        #print(x_target)
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        
        ### Forward f
        source_embeddings, source_outputs = model.forward_f(x_source)
        f_loss_source = f_criterion(nn.Softmax()(source_outputs), y_source)
        target_embeddings, target_outputs = model.forward_f(x_target)
        f_loss_target = f_criterion(nn.Softmax()(target_outputs), y_target)
        f_loss = f_loss_source + f_loss_target
        
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        fd_loss = d_criterion(source_dz, target_dz)
        
        total_f_loss = f_loss + d_criterion_w * fd_loss
        
        # update f 
        f_optimizer.zero_grad()
        total_f_loss.backward()
        f_optimizer.step()
        
        # log progress
        total_f_loss_epoch += total_f_loss.item()
        f_loss_epoch += f_loss.item()
        fd_loss_epoch += fd_loss.item()
        total_f_loss_steps.append(total_f_loss.item())
        f_loss_steps.append(f_loss.item())
        fd_loss_steps.append(fd_loss.item())
        
        #total_f_loss = total_f_loss.detach()
        
        
        ### Forward d
        x_source, _ = source_batch
        x_target, _ = target_batch
        x_source = x_source.detach()
        x_target = x_target.detach()
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        with torch.no_grad():
            source_embeddings, _ = model.forward_f(x_source)
            target_embeddings, _ = model.forward_f(x_target)
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        d_loss = - d_criterion(source_dz, target_dz)
        
        # update d
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # log progress
        d_loss_epoch += d_loss.item()
        d_loss_steps.append(d_loss.item())
        
        train_steps += 1
        
    return total_f_loss_epoch/train_steps, f_loss_epoch/train_steps, fd_loss_epoch/train_steps, d_loss_epoch/train_steps, total_f_loss_steps, f_loss_steps, fd_loss_steps, d_loss_steps


def da_train_epoch_2branches(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device
    ):
    
    model.train()
    train_steps = 0.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    d_loss_epoch = 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    d_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        x_source, y_source = source_batch
        x_target, y_target = target_batch
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        
        ### Forward f
        # forward source with no_grad()
        with torch.no_grad():
            source_embeddings, source_outputs = model.forward_source(x_source)
        f_loss_source = f_criterion(nn.Softmax()(source_outputs), y_source)
        # forward target with grad
        target_embeddings, target_outputs = model.forward_target(x_target)
        f_loss = f_loss_source 
        
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        fd_loss = d_criterion(source_dz, target_dz)
        
        total_f_loss = f_loss + d_criterion_w * fd_loss
        
        # update f 
        f_optimizer.zero_grad()
        total_f_loss.backward()
        f_optimizer.step()
        
        # log progress
        total_f_loss_epoch += total_f_loss.item()
        f_loss_epoch += f_loss.item()
        fd_loss_epoch += fd_loss.item()
        total_f_loss_steps.append(total_f_loss.item())
        f_loss_steps.append(f_loss.item())
        fd_loss_steps.append(fd_loss.item())
        
        #total_f_loss = total_f_loss.detach()
        
        
        ### Forward d
        x_source, _ = source_batch
        x_target, _ = target_batch
        x_source = x_source.detach()
        x_target = x_target.detach()
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        with torch.no_grad():
            source_embeddings, _ = model.forward_source(x_source)
            target_embeddings, _ = model.forward_target(x_target)
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        d_loss = - d_criterion(source_dz, target_dz)
        
        # update d
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # log progress
        d_loss_epoch += d_loss.item()
        d_loss_steps.append(d_loss.item())
        
        train_steps += 1
        
    return total_f_loss_epoch/train_steps, f_loss_epoch/train_steps, fd_loss_epoch/train_steps, d_loss_epoch/train_steps, total_f_loss_steps, f_loss_steps, fd_loss_steps, d_loss_steps


def da_train_epoch_2branches_fully_supervised(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device
    ):
    
    model.train()
    train_steps = 0.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    d_loss_epoch = 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    d_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        x_source, y_source = source_batch
        x_target, y_target = target_batch
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        
        ### Forward f
        # forward source with no_grad()
        with torch.no_grad():
            source_embeddings, source_outputs = model.forward_source(x_source)
        f_loss_source = f_criterion(nn.Softmax()(source_outputs), y_source)
        # forward target with grad
        target_embeddings, target_outputs = model.forward_target(x_target)
        f_loss_target = f_criterion(nn.Softmax()(target_outputs), y_target)
        f_loss = f_loss_source + f_loss_target 
        
        """z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        fd_loss = d_criterion(source_dz, target_dz)"""
        
        #total_f_loss = f_loss + d_criterion_w * fd_loss
        total_f_loss = f_loss
        
        # update f 
        f_optimizer.zero_grad()
        total_f_loss.backward()
        f_optimizer.step()
        
        # log progress
        total_f_loss_epoch += total_f_loss.item()
        f_loss_epoch += f_loss.item()
        #fd_loss_epoch += fd_loss.item()
        total_f_loss_steps.append(total_f_loss.item())
        f_loss_steps.append(f_loss.item())
        #fd_loss_steps.append(fd_loss.item())
        
        #total_f_loss = total_f_loss.detach()
        
        
        ### Forward d
        """x_source, _ = source_batch
        x_target, _ = target_batch
        x_source = x_source.detach()
        x_target = x_target.detach()
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        with torch.no_grad():
            source_embeddings, _ = model.forward_source(x_source)
            target_embeddings, _ = model.forward_target(x_target)
        z = torch.cat([source_embeddings, target_embeddings], 0)
        d_z = model.forward_d(z)
        source_dz = d_z[:source_embeddings.shape[0]]
        target_dz = d_z[source_embeddings.shape[0]:]
        d_loss = - d_criterion(source_dz, target_dz)
        
        # update d
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # log progress
        d_loss_epoch += d_loss.item()
        d_loss_steps.append(d_loss.item())"""
        
        train_steps += 1
        
    return total_f_loss_epoch/train_steps, f_loss_epoch/train_steps, fd_loss_epoch/train_steps, d_loss_epoch/train_steps, total_f_loss_steps, f_loss_steps, fd_loss_steps, d_loss_steps    


# REWORK: clean (?) version with gradient penalty and better losses tracking
def da_eval_epoch(
    model,
    loader,
    criterion,
    device, 
    double_encoder=True
):
    model.eval()
    epoch_acc = 0.0
    epoch_loss = 0.0
    steps = 0.0
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if double_encoder: _, outputs = model.forward_target(x)
            else: _, outputs = model.forward_f(x)
            
            # loss
            #loss = criterion(nn.Softmax()(outputs), y)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            
            # predict class
            predictions = nn.Softmax()(outputs).detach().cpu().numpy().argmax(1)
            y_pred += predictions.tolist()
            y_true += y.detach().cpu().numpy().tolist()
            
            steps += 1
            
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_loss = epoch_loss / steps
    
    return epoch_acc, epoch_loss


def compute_confusion_matrix(
    model,
    source_loader,
    target_loader,
    device,
    output_name='confusion_matrix',
    double_encoder=False
): 
    model.eval()
    
    all_source_preds = []
    all_target_preds = []
    all_source_labels = []
    all_target_labels = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for source_batch, target_batch in zip(source_loader, target_loader):
            x_source, y_source = source_batch
            x_target, y_target = target_batch
            x_source, y_source = x_source.to(device), y_source.to(device)
            x_target, y_target = x_target.to(device), y_target.to(device)
            
            _, outputs_source = model.forward_f(x_source)
            _, outputs_target = model.forward_f(x_target)
            outputs_source = nn.Softmax()(outputs_source)
            outputs_target = nn.Softmax()(outputs_target)
            source_preds = torch.argmax(outputs_source, dim=1)
            target_preds = torch.argmax(outputs_target, dim=1)
            
            all_source_preds.append(source_preds)
            all_target_preds.append(target_preds)
            all_source_labels.append(y_source)
            all_target_labels.append(y_target)
            all_preds.append(source_preds)
            all_preds.append(target_preds)
            all_labels.append(y_source)
            all_labels.append(y_target)
    
    # concat all batches
    all_source_preds = torch.cat(all_source_preds).cpu().numpy()
    all_target_preds = torch.cat(all_target_preds).cpu().numpy()
    all_source_labels = torch.cat(all_source_labels).cpu().numpy()
    all_target_labels = torch.cat(all_target_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    
    # compute cm
    cm_source = confusion_matrix(all_source_labels, all_source_preds)
    cm_target = confusion_matrix(all_target_labels, all_target_preds)
    cm_all = confusion_matrix(all_labels, all_preds)
    
    # save fig
    plt.figure()
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix source+target')
    plt.savefig(f'figures/confusion_matrix/all/{output_name}_all.png')
    plt.close()
    
    plt.figure()
    sns.heatmap(cm_source, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix source')
    plt.savefig(f'figures/confusion_matrix/source/{output_name}_source.png')
    plt.close()
    
    plt.figure()
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix target')
    plt.savefig(f'figures/confusion_matrix/target/{output_name}_target.png')
    plt.close()
    
    
    
    

def soft_relu(x):
    return ((-x.abs()).exp() + 1.0).log() + torch.clip(x, min=0.0)

def wasserstein(d1, d2):
    #part1 = torch.mean(d1)
    #part2 = torch.mean(d2)
    
    part1 = torch.mean(soft_relu(d1))
    part2 = torch.mean(soft_relu(d2))
    return part1, part2

def build_d_grad_loss(model, z1, z2, device):
    #gradient penalty with interpolations
    alpha = torch.rand(z1.shape[0], 1, device=device)
    z_intp = (alpha * z1 + (1-alpha) * z2).detach()
    z_intp.requires_grad = True
    dz_intp = model.forward_d(z_intp)
    z_intp_grad = autograd.grad(outputs=dz_intp, inputs=z_intp, grad_outputs=torch.ones(dz_intp.shape, device=device), create_graph=True, retain_graph=True)[0]
    z_grad_norm = (z_intp_grad.square().sum(-1) + 1e-10).sqrt()
    d_grad_loss = (z_grad_norm - 1.0).square().mean()
    return d_grad_loss

def build_d_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, double_encoder=True):
    x_source, _ = source_batch
    x_target, _ = target_batch
    x_source = x_source.to(device)
    x_target = x_target.to(device)
    """with torch.no_grad():
        if double_encoder:
            source_embeddings, _ = model.forward_source(x_source)
            target_embeddings, _ = model.forward_target(x_target)
        else: 
            source_embeddings, _ = model.forward_f(x_source)
            target_embeddings, _ = model.forward_f(x_target)
            """
    if double_encoder:
        source_embeddings, _ = model.forward_source(x_source)
        target_embeddings, _ = model.forward_target(x_target)
    else: 
        source_embeddings, _ = model.forward_f(x_source)
        target_embeddings, _ = model.forward_f(x_target)
            
    z = torch.cat([source_embeddings, target_embeddings], 0)
    d_z = model.forward_d(z)
    source_dz = d_z[:source_embeddings.shape[0]]
    target_dz = d_z[source_embeddings.shape[0]:]
    d_source_loss, d_target_loss = d_criterion(source_dz, target_dz)
    #d_loss = d_source_loss - d_target_loss
    d_loss = - d_source_loss + d_target_loss
    # gradient penalty
    d_grad_loss = build_d_grad_loss(model, source_embeddings, target_embeddings, device)
    total_d_loss = d_loss + 10.0 * d_grad_loss
    return total_d_loss, d_loss, d_source_loss, d_target_loss, d_grad_loss

def build_f_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, d_criterion_w= 1.0, supervised=False, double_encoder=True):
    x_source, y_source = source_batch
    x_target, y_target = target_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
    #print(y_source)
    #print(y_target)
    
    if double_encoder:
        # forward source with no_grad()
        with torch.no_grad():
            source_embeddings, source_outputs = model.forward_source(x_source)
        f_loss_source = f_criterion(nn.Softmax()(source_outputs), y_source)
        #f_loss_source = f_criterion(source_outputs, y_source)
        # forward target with grad
        target_embeddings, target_outputs = model.forward_target(x_target)
        
        if supervised:
            f_loss_target = f_criterion(nn.Softmax()(target_outputs), y_target)
            #f_loss_target = f_criterion(target_outputs, y_target)
            f_loss = f_loss_target
            
    else: 
        # forward source
        source_embeddings, source_outputs = model.forward_f(x_source)
        #f_loss_source = f_criterion(nn.Softmax()(source_outputs), y_source)
        f_loss_source = f_criterion(source_outputs, y_source)
        # forward target
        target_embeddings, target_outputs = model.forward_f(x_target)
        if supervised:
            #f_loss_target = f_criterion(nn.Softmax()(target_outputs), y_target)
            f_loss_target = f_criterion(target_outputs, y_target)
            f_loss = f_loss_source + f_loss_target
    
    z = torch.cat([source_embeddings, target_embeddings], 0)
    d_z = model.forward_d(z)
    source_dz = d_z[:source_embeddings.shape[0]]
    target_dz = d_z[source_embeddings.shape[0]:]
    fd_source_loss, fd_target_loss = d_criterion(source_dz, target_dz)
    #fd_loss = - fd_source_loss
    fd_loss = - fd_target_loss
    #fd_loss = -fd_target_loss + fd_source_loss
    
    if supervised: 
        total_f_loss = f_loss + (d_criterion_w * fd_loss)
        #total_f_loss = d_criterion_w * f_loss + (1 - d_criterion_w)*fd_loss
        return total_f_loss, f_loss, fd_loss

    elif not supervised and not double_encoder:
        total_f_loss = f_loss_source + (d_criterion_w * fd_loss)
        return total_f_loss
    
    elif not supervised and double_encoder:
        total_f_loss = fd_loss
        return total_f_loss
    
    
def da_train_epoch_2branches_v2(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        n_critic=5
    ):
    
    model.train()
    train_steps = 1.0
    total_f_loss_epoch = 0.0
    total_d_loss_epoch, d_loss_epoch, d_source_loss_epoch, d_target_loss_epoch, d_grad_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_f_loss_steps = []
    total_d_loss_steps, d_loss_steps, d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps = [], [], [], [], []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # train D
        total_d_loss, d_loss, d_source_loss, d_target_loss, d_grad_loss = build_d_loss(model, source_batch, target_batch, f_criterion, d_criterion, device)
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
        
        # train E
        if train_steps % n_critic == 0:
            total_f_loss = build_f_loss(model, source_batch, target_batch, f_criterion, d_criterion, device)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]


def da_train_epoch_2branches_fully_supervised_v2(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        n_critic=5
    ):
    
    model.train()
    train_steps = 1.0
    total_f_loss_epoch, f_loss_epoch, fd_loss_epoch = 0.0, 0.0, 0.0
    total_d_loss_epoch, d_loss_epoch, d_source_loss_epoch, d_target_loss_epoch, d_grad_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_f_loss_steps, f_loss_steps, fd_loss_steps = [], [], []
    total_d_loss_steps, d_loss_steps, d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps = [], [], [], [], []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # train D
        total_d_loss, d_loss, d_source_loss, d_target_loss, d_grad_loss = build_d_loss(model, source_batch, target_batch, f_criterion, d_criterion, device)
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
        
        # train E
        if train_steps % n_critic == 0:
            total_f_loss, f_loss, fd_source_loss = build_f_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, d_criterion_w=d_criterion_w, supervised=True)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            f_loss_epoch += f_loss.item()
            fd_loss_epoch += fd_source_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
            f_loss_steps.append(f_loss.item())
            fd_loss_steps.append(fd_source_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic), f_loss_epoch/(train_steps/n_critic), fd_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, f_loss_steps, fd_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]


def da_train_epoch_1branch_fully_supervised_v2(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        n_critic=5
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
        
        # train E
        if train_steps % n_critic == 0:
            total_f_loss, f_loss, fd_source_loss = build_f_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, d_criterion_w=d_criterion_w, supervised=True, double_encoder=False)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            f_loss_epoch += f_loss.item()
            fd_loss_epoch += fd_source_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
            f_loss_steps.append(f_loss.item())
            fd_loss_steps.append(fd_source_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic), f_loss_epoch/(train_steps/n_critic), fd_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, f_loss_steps, fd_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]

def da_train_epoch_1branch_v2(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        f_optimizer,
        d_optimizer,
        d_criterion_w,
        device, 
        n_critic=5
    ):
    
    model.train()
    train_steps = 1.0
    total_f_loss_epoch = 0.0
    total_d_loss_epoch, d_loss_epoch, d_source_loss_epoch, d_target_loss_epoch, d_grad_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_f_loss_steps = []
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
        
        # train E
        if train_steps % n_critic == 0:
            total_f_loss = build_f_loss(model, source_batch, target_batch, f_criterion, d_criterion, device, double_encoder=False, supervised=False)
            f_optimizer.zero_grad()
            total_f_loss.backward()
            f_optimizer.step()
                
            # F logs
            total_f_loss_epoch += total_f_loss.item()
            total_f_loss_steps.append(total_f_loss.item())
        
        train_steps += 1
        
    return [total_f_loss_epoch/(train_steps/n_critic),
            total_d_loss_epoch/train_steps, d_loss_epoch/train_steps,
            d_source_loss_epoch/train_steps, d_target_loss_epoch/train_steps, d_grad_loss_epoch/train_steps,
            total_f_loss_steps, 
            total_d_loss_steps, d_loss_steps,
            d_source_loss_steps, d_target_loss_steps, d_grad_loss_steps]
