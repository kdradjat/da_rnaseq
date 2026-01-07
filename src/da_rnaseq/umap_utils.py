import random
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from umap import umap_ as mp
#import umap


def latent_space_viz(model, source_loader, target_loader, epoch, device, name="model", saved_umap=None, double_encoder=True):
    np.random.seed(42)
    
    # Collect data from data loaders
    source_latent_space_data = []
    source_labels = []
    target_latent_space_data = []
    target_labels = []
    
    model.eval()
    with torch.no_grad():
        for source_batch in source_loader:
            x_source, y_source = source_batch
            x_source = x_source.to(device)
            if double_encoder: z_source, _ = model.forward_source(x_source)
            else: z_source, _ = model.forward_f(x_source)
            z_source = z_source.detach().cpu()
            source_latent_space_data.append(z_source.numpy())
            source_labels.append(y_source.numpy())

        for target_batch in target_loader:
            x_target, y_target = target_batch
            x_target = x_target.to(device)
            if double_encoder: z_target, _ = model.forward_target(x_target)
            else: z_target, _ = model.forward_f(x_target)
            z_target = z_target.detach().cpu()
            target_latent_space_data.append(z_target.numpy())
            target_labels.append(y_target.numpy())
    
    source_latent_space_data = np.concatenate(source_latent_space_data, axis=0)
    source_labels = np.concatenate(source_labels, axis=0)
    #source_domain_labels = np.zeros(source_latent_space_data.shape[0])
    source_domain_labels = np.array(["r" for i in range(source_latent_space_data.shape[0])])
    target_latent_space_data = np.concatenate(target_latent_space_data, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)
    #target_domain_labels = np.ones(target_latent_space_data.shape[0])
    target_domain_labels = np.array(["b" for i in range(target_latent_space_data.shape[0])])
    
    latent_space_data = np.concatenate([source_latent_space_data, target_latent_space_data], axis=0)
    labels = np.concatenate([source_labels, target_labels], axis=0)
    domain_labels = np.concatenate([source_domain_labels, target_domain_labels], axis=0)
    
    #Apply UMAP to source only
    # Initialize umap
    if saved_umap == None:
        umap_model = mp.UMAP(n_components=2, n_neighbors=15, random_state=42, n_jobs=1)
        #umap_results = umap_model.fit_transform(latent_space_data)
        fixed_umap = umap_model.fit(source_latent_space_data)
    else: fixed_umap = saved_umap
    
    umap_source = fixed_umap.transform(source_latent_space_data)
    umap_target = fixed_umap.transform(target_latent_space_data)
    
    # vizualize
    plt.figure()
    #plt.scatter(umap_results[:,0], umap_results[:,1], c=domain_labels, s=0.5)
    plt.scatter(umap_source[:,0], umap_source[:,1], c='r', label='source', s=0.5)
    plt.scatter(umap_target[:,0], umap_target[:,1], c='b', label='target', s=0.5)
    plt.title(f"UMAP projection of Latent Space with Domain Labels (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    #plt.colorbar(label='Domain Labels')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.savefig(f'figures/umap/{name}_umap_domain_epoch{epoch}.png')
    
    plt.figure()
    #plt.scatter(umap_results[:,0], umap_results[:,1], c=domain_labels, s=0.5)
    plt.scatter(umap_source[:,0], umap_source[:,1], c=source_labels, label='source', s=0.5)
    plt.scatter(umap_target[:,0], umap_target[:,1], c=target_labels, label='target', s=0.5)
    plt.title(f"UMAP projection of Latent Space with Class Labels (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    #plt.colorbar(label='Domain Labels')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.savefig(f'figures/umap/{name}_umap_class_epoch{epoch}.png')
    
    if not double_encoder: return None
    
    return fixed_umap


def latent_space_viz_flex(model, source_loader, target_loader, epoch, device, name="model", saved_umap=None, double_encoder=True):
    np.random.seed(42)
    
    # Collect data from data loaders
    source_latent_space_data = []
    source_labels = []
    target_latent_space_data = []
    target_labels = []
    
    model.eval()
    with torch.no_grad():
        for source_batch in source_loader:
            x_source, y_source = source_batch
            x_source = x_source.to(device)
            if double_encoder: z_source, _ = model.forward_source(x_source)
            else: z_source, _ = model.forward_f(x_source)
            z_source = z_source.detach().cpu()
            source_latent_space_data.append(z_source.numpy())
            source_labels.append(y_source.numpy())

        for target_batch in target_loader:
            x_target, y_target = target_batch
            x_target = x_target.to(device)
            if double_encoder: z_target, _ = model.forward_target(x_target)
            else: z_target, _ = model.forward_f(x_target)
            z_target = z_target.detach().cpu()
            target_latent_space_data.append(z_target.numpy())
            target_labels.append(y_target.numpy())
    
    source_latent_space_data = np.concatenate(source_latent_space_data, axis=0)
    source_labels = np.concatenate(source_labels, axis=0)
    #source_domain_labels = np.zeros(source_latent_space_data.shape[0])
    source_domain_labels = np.array(["r" for i in range(source_latent_space_data.shape[0])])
    target_latent_space_data = np.concatenate(target_latent_space_data, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)
    #target_domain_labels = np.ones(target_latent_space_data.shape[0])
    target_domain_labels = np.array(["b" for i in range(target_latent_space_data.shape[0])])
    
    latent_space_data = np.concatenate([source_latent_space_data, target_latent_space_data], axis=0)
    labels = np.concatenate([source_labels, target_labels], axis=0)
    domain_labels = np.concatenate([source_domain_labels, target_domain_labels], axis=0)
    
    #Apply UMAP to source only
    # Initialize umap
    #umap_model = mp.UMAP(n_components=2, n_neighbors=15, random_state=42, n_jobs=1)
    #umap_model = mp.UMAP(n_components=2, n_neighbors=15, random_state=42)
    umap_model = mp.UMAP(n_components=2, n_neighbors=15)
    #umap_results = umap_model.fit_transform(latent_space_data)
    fixed_umap = umap_model.fit(latent_space_data)
    
    #umap_source = fixed_umap.transform(source_latent_space_data)
    #umap_target = fixed_umap.transform(target_latent_space_data)
    #umap_source = fixed_umap.fit_transform(source_latent_space_data)
    #umap_target = fixed_umap.fit_transform(target_latent_space_data)
    
    #umap_all = fixed_umap.transform(latent_space_data)
    
    umap_all = umap_model.fit_transform(latent_space_data)
    umap_source = umap_all[:len(source_latent_space_data)]
    umap_target = umap_all[len(source_latent_space_data):]
    
    # vizualize
    plt.figure()
    plt.scatter(umap_all[:,0], umap_all[:,1], c=domain_labels, label='source', s=0.3)
    plt.title(f"UMAP projection of Latent Space with Domain Labels (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(f'figures/umap/{name}_umap_domain_epoch{epoch}_flex.png')
    plt.figure()
    
    plt.scatter(umap_all[:,0], umap_all[:,1], c=labels, label='source', s=0.3)
    plt.title(f"UMAP projection of Latent Space with Class Labels (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(f'figures/umap/{name}_umap_class_epoch{epoch}_flex.png')
    
    plt.figure()
    #plt.scatter(umap_results[:,0], umap_results[:,1], c=domain_labels, s=0.5)
    plt.scatter(umap_source[:,0], umap_source[:,1], c=source_labels, label='source', s=0.5)
    plt.title(f"UMAP projection of Latent Space Source Data (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    #plt.colorbar(label='Domain Labels')
    #plt.xlim(-20, 20)
    #plt.ylim(-20, 20)
    plt.savefig(f'figures/umap/{name}_umap_source_epoch{epoch}.png')
    
    plt.figure()
    plt.scatter(umap_target[:,0], umap_target[:,1], c=target_labels, label='target', s=0.5)
    plt.title(f"UMAP projection of Latent Space Target Data (Epoch {epoch})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    #plt.colorbar(label='Domain Labels')
    #plt.xlim(-20, 20)
    #plt.ylim(-20, 20)
    plt.savefig(f'figures/umap/{name}_umap_target_epoch{epoch}.png')
    
    if not double_encoder: return None
    
    return fixed_umap

    
