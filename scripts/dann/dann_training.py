import sys, os
import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, TensorDataset

from da_rnaseq.data import *
from da_rnaseq.models import *
from da_rnaseq.utils import *
from da_rnaseq.dann_utils import *
from da_rnaseq.umap_utils import *

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--source_data_file",
        default=None,
        type=str,
        required=True,
        help="Source data file"
    )
    parser.add_argument(
        "--source_label_file",
        default=None,
        type=str,
        required=True,
        help="Source label file"
    )
    parser.add_argument(
        "--target_data_file",
        default=None,
        type=str,
        required=True,
        help="Target data file"
    )
    parser.add_argument(
        "--target_label_file",
        default=None,
        type=str,
        required=True,
        help="Target label file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=50,
        help="number of epoch"
    )
    parser.add_argument(
        "--supervised",
        action='store_true',
        help='for fully supervised case'
    )
    parser.add_argument(
        "--plot_umap",
        action='store_true',
        help='visualization of latent space evert epoch'
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default='model',
        help='name to save umap plot'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='da_model',
        help='name to save model'
    )
    parser.add_argument(
        "--w_d_criterion",
        type=float,
        default=1.0,
        help='contribution of Discriminator loss in Encoder loss'
    )
    parser.add_argument(
        "--w_progressive",
        action='store_true',
        help='decrese discriminator contribution during the training'
    )
    parser.add_argument(
        "--early_stop",
        action='store_true',
        help='stop training with early stop'
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help='dropout rate'
    )
    parser.add_argument(
        "--binary",
        action='store_true',
        help='binary cancer/non-cancer classification'
    )
    
    args = parser.parse_args()
    
    # param
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='min')
    
    ### data preprocessing
    source = read_process_data_ARCHS4(args.source_data_file, args.source_label_file, binary=args.binary)
    data_source, source_labels = source[:,1:], source[:,0]
    #target = read_process_data_TCGA(args.target_data_file,args.target_label_file, replace=True, binary=args.binary)
    #data_target, target_labels = target[:,1:], target[:,0]
    target = read_process_data_gtex(args.target_data_file,args.target_label_file)
    data_target, target_labels = target[:,1:], target[:,0]
    
    # split
    x_train_source, x_valid_source, y_train_source, y_valid_source = train_test_split(data_source, source_labels, test_size=0.1, stratify=source_labels, random_state=42)
    x_train_target, x_valid_target, y_train_target, y_valid_target = train_test_split(data_target, target_labels, test_size=0.1, stratify=target_labels, random_state=42)
    
    # into datasets
    source_train_dataset = TensorDataset(torch.Tensor(x_train_source), torch.LongTensor(y_train_source))
    source_valid_dataset = TensorDataset(torch.Tensor(x_valid_source), torch.LongTensor(y_valid_source))
    target_train_dataset = TensorDataset(torch.Tensor(x_train_target), torch.LongTensor(y_train_target))
    target_valid_dataset = TensorDataset(torch.Tensor(x_valid_target), torch.LongTensor(y_valid_target))
    
    # compute sampling weights
    num_source_train, num_target_train = len(x_train_source), len(x_train_target) 
    num_source_valid, num_target_valid = len(x_valid_source), len(x_valid_target)
    source_train_weights = [1.0/num_source_train] * num_source_train
    target_train_weights = [1.0/num_target_train] * num_target_train
    source_valid_weights = [1.0/num_source_valid] * num_source_valid
    target_valid_weights = [1.0/num_target_valid] * num_target_valid
    # create samplers 
    source_train_sampler = WeightedRandomSampler(source_train_weights, num_samples=num_source_train, replacement=True)
    target_train_sampler = WeightedRandomSampler(target_train_weights, num_samples=num_source_train, replacement=True)  # Match source size
    source_valid_sampler = WeightedRandomSampler(source_valid_weights, num_samples=num_source_valid, replacement=True)
    target_valid_sampler = WeightedRandomSampler(target_valid_weights, num_samples=num_source_valid, replacement=True)  # Match source size
    
    # into DataLoaders
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, sampler=source_train_sampler)
    source_valid_loader = DataLoader(source_valid_dataset, batch_size=args.batch_size, sampler=source_valid_sampler)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, sampler=target_train_sampler)
    target_valid_loader = DataLoader(target_valid_dataset, batch_size=args.batch_size, sampler=target_valid_sampler)

    source_train_loader_orig = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=False)
    source_valid_loader_orig = DataLoader(source_valid_dataset, batch_size=args.batch_size, shuffle=False)
    target_train_loader_orig = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=False)
    target_valid_loader_orig = DataLoader(target_valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # models parameters
    e_neuron_list = [256,256,256]
    c_neuron_list = [128, 64]
    d_neuron_list = [256,128,64]
    
    # wd weight parameter
    wd_list = [args.w_d_criterion for i in range(args.nb_epochs)]
    if args.w_progressive:
        wd_list = [0 for i in range(args.nb_epochs)]
        wd_list[:4] = [1.0, 0.75, 0.5, 0.25]
    
    ### Build model
    model = DANN_Model(
        input_dim=x_train_source.shape[1],
        nb_classes=len(np.unique(y_train_source)),
        e_neuron_list=e_neuron_list,
        c_neuron_list=c_neuron_list,
        d_neuron_list=d_neuron_list,
        dropout=args.dropout
    ).to(device)
    
    ### Training
    epochs = args.nb_epochs
    # optimizers
    params = model.parameters()
    optimizer = AdamW(params, lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-1)
    # losses
    f_loss_function = torch.nn.CrossEntropyLoss()
    d_loss_function = torch.nn.BCELoss()
    
    # logs lists
    train_source_acc_history, train_source_loss_history = [], []
    valid_source_acc_history, valid_source_loss_history = [], []
    train_target_acc_history, train_target_loss_history = [], []
    valid_target_acc_history, valid_target_loss_history = [], []
    
    total_loss_history, f_loss_history, d_loss_history = [], [], []
    
    total_loss_steps_history, f_loss_steps_history, d_loss_steps_history = [], [], []
    
    start = time.time()
    saved_umap=None
    
    # D weight index
    wd_index = 0
    for epoch in range(1, epochs+1):    
        # visualization of latent space
        if args.plot_umap:
            #saved_umap_trans = latent_space_viz(model, source_valid_loader_orig, target_train_loader_orig, epoch, device=device, name=args.plot_name, saved_umap=saved_umap, double_encoder=False)
            saved_umap_trans = latent_space_viz_flex(model, source_train_loader_orig, target_train_loader_orig, epoch, device=device, name=f'{args.plot_name}_train', saved_umap=saved_umap, double_encoder=False)
            saved_umap_trans = latent_space_viz_flex(model, source_valid_loader_orig, target_valid_loader_orig, epoch, device=device, name=f'{args.plot_name}_valid', saved_umap=saved_umap, double_encoder=False)
            saved_umap = saved_umap_trans
        # Supervised
        if args.supervised:
            losses_logs = da_train_epoch_1branch_dann(
                model, 
                source_train_loader, 
                target_train_loader, 
                f_loss_function,
                d_loss_function,
                optimizer,
                device=device
            )
        # Unsupervised
        else:
            losses_logs = da_train_epoch_1branch_dann_unsupervised(
                model, 
                source_train_loader, 
                target_train_loader, 
                f_loss_function,
                d_loss_function,
                optimizer,
                device=device
            )
        
        # eval on train source/target
        train_source_acc, train_source_loss = da_eval_epoch(model, source_train_loader_orig, f_loss_function, device, double_encoder=False)
        train_target_acc, train_target_loss = da_eval_epoch(model, target_train_loader_orig, f_loss_function, device, double_encoder=False)
        
        # eval on valid source/target
        valid_source_acc, valid_source_loss = da_eval_epoch(model, source_valid_loader_orig, f_loss_function, device, double_encoder=False)
        valid_target_acc, valid_target_loss = da_eval_epoch(model, target_valid_loader_orig, f_loss_function, device, double_encoder=False)
        
        # update logs
        train_source_acc_history.append(train_source_acc)
        train_source_loss_history.append(train_source_loss)
        train_target_acc_history.append(train_target_acc)
        train_target_loss_history.append(train_target_loss)
        valid_source_acc_history.append(valid_source_acc)
        valid_source_loss_history.append(valid_source_loss)
        valid_target_acc_history.append(valid_target_acc)
        valid_target_loss_history.append(valid_target_loss)
        
        if args.supervised:
            total_loss_history.append(losses_logs[0])
            d_loss_history.append(losses_logs[1])
            f_loss_history.append(losses_logs[2])
            
            total_loss_steps_history += losses_logs[3]
            d_loss_steps_history += losses_logs[4]
            f_loss_steps_history += losses_logs[5]
            
        else:
            total_loss_history.append(losses_logs[0])
            d_loss_history.append(losses_logs[1])
            f_loss_history.append(losses_logs[2])
            
            total_loss_steps_history += losses_logs[3]
            d_loss_steps_history += losses_logs[4]
            f_loss_steps_history += losses_logs[5]
        
        if early_stopping(valid_target_loss) and args.early_stop:
            print(f"Early stopping triggered. Stopping training at epoch {epoch}")
            print(f'Epoch {epoch}/{epochs} : loss = {valid_target_loss_history[-1]}')
            torch.save(model.state_dict(), f'models/{args.model_name}.pt')
            break
        
        # save model
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} : loss = {total_loss_history[-1]}')
            torch.save(model.state_dict(), f'models/{args.model_name}.pt')
            
        wd_index += 1
            
    end = time.time()
    print(f'Training time: {end-start}')
    
    # save logs
    
    if args.supervised:
        # df epochs
        df_epoch = pd.DataFrame(columns=['epoch',
                                   'total_loss',
                                   'train_f_loss',
                                   'train_d_loss',
                                   'train_source_acc',
                                   'train_source_loss',
                                   'train_target_acc',
                                   'train_target_loss',
                                   'valid_source_acc',
                                   'valid_source_loss',
                                   'valid_target_acc',
                                   'valid_target_loss'
                                  ])
        df_epoch['epoch'] = [i for i in range(len(total_loss_history))]
        df_epoch['total_loss'] = total_loss_history
        df_epoch['train_f_loss'] = f_loss_history
        df_epoch['train_d_loss'] = d_loss_history
        df_epoch['train_source_acc'] = train_source_acc_history
        df_epoch['train_source_loss'] = train_source_loss_history
        df_epoch['train_target_acc'] = train_target_acc_history
        df_epoch['train_target_loss'] = train_target_loss_history
        df_epoch['valid_source_acc'] = valid_source_acc_history
        df_epoch['valid_source_loss'] = valid_source_loss_history
        df_epoch['valid_target_acc'] = valid_target_acc_history
        df_epoch['valid_target_loss'] = valid_target_loss_history
        df_epoch.to_csv(f'training_csv/training_history_{args.model_name}.csv')

        # df steps 
        df_encoder_steps = pd.DataFrame(columns=['step',
                                   'total_f_loss',
                                   'f_loss',
                                   'd_loss'
                                  ])
        df_encoder_steps['step'] = [i for i in range(len(total_loss_steps_history))]
        df_encoder_steps['total_f_loss'] = total_loss_steps_history
        df_encoder_steps['f_loss'] = f_loss_steps_history
        df_encoder_steps['d_loss'] = d_loss_steps_history
        df_encoder_steps.to_csv(f'training_csv/encoder_training_history_steps_{args.model_name}.csv')
        
    else:
        # df epochs
        df_epoch = pd.DataFrame(columns=['epoch',
                                   'total_loss',
                                   'train_f_loss',
                                   'train_d_loss',
                                   'train_source_acc',
                                   'train_source_loss',
                                   'train_target_acc',
                                   'train_target_loss',
                                   'valid_source_acc',
                                   'valid_source_loss',
                                   'valid_target_acc',
                                   'valid_target_loss'
                                  ])
        df_epoch['epoch'] = [i for i in range(len(total_loss_history))]
        df_epoch['total_loss'] = total_loss_history
        df_epoch['train_f_loss'] = f_loss_history
        df_epoch['train_d_loss'] = d_loss_history
        df_epoch['train_source_acc'] = train_source_acc_history
        df_epoch['train_source_loss'] = train_source_loss_history
        df_epoch['train_target_acc'] = train_target_acc_history
        df_epoch['train_target_loss'] = train_target_loss_history
        df_epoch['valid_source_acc'] = valid_source_acc_history
        df_epoch['valid_source_loss'] = valid_source_loss_history
        df_epoch['valid_target_acc'] = valid_target_acc_history
        df_epoch['valid_target_loss'] = valid_target_loss_history
        df_epoch.to_csv(f'training_csv/training_history_{args.model_name}.csv')

        # df steps 
        df_encoder_steps = pd.DataFrame(columns=['step',
                                   'total_f_loss',
                                   'f_loss',
                                   'd_loss'
                                  ])
        df_encoder_steps['step'] = [i for i in range(len(total_loss_steps_history))]
        df_encoder_steps['total_f_loss'] = total_loss_steps_history
        df_encoder_steps['f_loss'] = f_loss_steps_history
        df_encoder_steps['d_loss'] = d_loss_steps_history
        df_encoder_steps.to_csv(f'training_csv/encoder_training_history_steps_{args.model_name}.csv')
    
if __name__ == "__main__":
    main()
        
        
    
    
