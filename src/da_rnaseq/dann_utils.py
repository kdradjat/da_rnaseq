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


def da_train_epoch_1branch_dann(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        optimizer,
        device
    ):
    
    model.train()
    train_steps = 1.0
    f_loss_epoch = 0.0
    d_loss_epoch = 0.0
    total_loss_epoch = 0.0
    
    f_loss_steps = []
    d_loss_steps = []
    total_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # training
        f_loss, d_loss, total_loss = build_loss_dann(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=True)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # logs
        total_loss_epoch += total_loss.item()
        d_loss_epoch += d_loss.item()
        f_loss_epoch += f_loss.item()
        
        total_loss_steps.append(total_loss.item())
        d_loss_steps.append(d_loss.item())
        f_loss_steps.append(f_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps, d_loss_epoch/train_steps, f_loss_epoch/train_steps,
            total_loss_steps, d_loss_steps, f_loss_steps]


def da_train_epoch_1branch_dann_unsupervised(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        optimizer,
        device
    ):
    
    model.train()
    train_steps = 1.0
    f_loss_epoch = 0.0
    d_loss_epoch = 0.0
    total_loss_epoch = 0.0
    
    f_loss_steps = []
    d_loss_steps = []
    total_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # training
        f_loss, d_loss, total_loss = build_loss_dann(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=False)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # logs
        total_loss_epoch += total_loss.item()
        d_loss_epoch += d_loss.item()
        f_loss_epoch += f_loss.item()
        
        total_loss_steps.append(total_loss.item())
        d_loss_steps.append(d_loss.item())
        f_loss_steps.append(f_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps, d_loss_epoch/train_steps, f_loss_epoch/train_steps,
            total_loss_steps, d_loss_steps, f_loss_steps]
        

def build_loss_dann(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=True):
    x_source, y_source = source_batch
    x_target, y_target = target_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
     
    x = torch.cat([x_source, x_target], dim=0)
    if supervised: y = torch.cat([y_source, y_target], dim=0)
    domain_labels = torch.cat([torch.ones(len(x_source), 1), torch.zeros(len(x_target), 1)]).to(device)
    
    # forward
    embeddings, f_outputs, d_outputs = model.forward(x)
    
    # label prediction
    if supervised: 
        #f_loss = f_criterion(f_outputs, y)
        f_loss = f_criterion(nn.Softmax()(f_outputs), y)
    else: 
        #f_loss = f_criterion(f_outputs[:len(x_source)], y_source)
        f_loss = f_criterion(nn.Softmax()(f_outputs[:len(x_source)]), y_source)
    
    # domain predictor
    d_loss = d_criterion(d_outputs, domain_labels)
    
    total_loss = f_loss + d_loss
    
    return f_loss, d_loss, total_loss

### 2 branches ###

def da_train_epoch_2branches_dann(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        optimizer,
        device
    ):
    
    model.train()
    train_steps = 1.0
    f_loss_epoch = 0.0
    d_loss_epoch = 0.0
    total_loss_epoch = 0.0
    
    f_loss_steps = []
    d_loss_steps = []
    total_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # training
        f_loss, d_loss, total_loss = build_loss_dann_2branches(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=True)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # logs
        total_loss_epoch += total_loss.item()
        d_loss_epoch += d_loss.item()
        f_loss_epoch += f_loss.item()
        
        total_loss_steps.append(total_loss.item())
        d_loss_steps.append(d_loss.item())
        f_loss_steps.append(f_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps, d_loss_epoch/train_steps, f_loss_epoch/train_steps,
            total_loss_steps, d_loss_steps, f_loss_steps]


def da_train_epoch_2branches_dann_unsupervised(
        model, 
        source_loader, 
        target_loader, 
        f_criterion,
        d_criterion,
        optimizer,
        device
    ):
    
    model.train()
    train_steps = 1.0
    f_loss_epoch = 0.0
    d_loss_epoch = 0.0
    total_loss_epoch = 0.0
    
    f_loss_steps = []
    d_loss_steps = []
    total_loss_steps = []
    
    for source_batch, target_batch in zip(source_loader, target_loader):
        # training
        f_loss, d_loss, total_loss = build_loss_dann_2branches(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=False)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # logs
        total_loss_epoch += total_loss.item()
        d_loss_epoch += d_loss.item()
        f_loss_epoch += f_loss.item()
        
        total_loss_steps.append(total_loss.item())
        d_loss_steps.append(d_loss.item())
        f_loss_steps.append(f_loss.item())
        
        train_steps += 1
        
    return [total_loss_epoch/train_steps, d_loss_epoch/train_steps, f_loss_epoch/train_steps,
            total_loss_steps, d_loss_steps, f_loss_steps]




def build_loss_dann_2branches(model, source_batch, target_batch, f_criterion, d_criterion, device, supervised=True):
    x_source, y_source = source_batch
    x_target, y_target = target_batch
    x_source = x_source.to(device)
    y_source = y_source.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)
     
    x = torch.cat([x_source, x_target], dim=0)
    if supervised: y = torch.cat([y_source, y_target], dim=0)
    domain_labels = torch.cat([torch.ones(len(x_source), 1), torch.zeros(len(x_target), 1)]).to(device)
    
    # forward
    embeddings_source, outputs_source = model.forward_source(x_source)
    embeddings_target, outputs_target = model.forward_target(x_target)
    #embeddings, f_outputs, d_outputs = model.forward(x)
    d_outputs_source = model.forward_d(embeddings_source)
    d_outputs_target = model.forward_d(embeddings_target)
    
    # label prediction
    f_loss = f_criterion(outputs_target, y_target)
    
    # domain predictor
    d_outputs = torch.cat([d_outputs_source, d_outputs_target], 0)
    d_loss = d_criterion(d_outputs, domain_labels)
    
    if supervised: 
        total_loss = f_loss + d_loss
    else: 
        total_loss = d_loss
    
    return f_loss, d_loss, total_loss
    
    