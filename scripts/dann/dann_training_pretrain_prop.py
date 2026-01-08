import sys, os
import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW, SGD
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
        "--test_data_file",
        default=None,
        type=str,
        required=True,
        help="Test data file"
    )
    parser.add_argument(
        "--test_label_file",
        default=None,
        type=str,
        required=True,
        help="Test label file"
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
        "--w_d_criterion",
        type=float,
        default=1.0,
        help='contribution of Discriminator loss in Encoder loss'
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default='model',
        help='name of output file'
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help='dropout rate'
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help='number of patience epoch for early stop'
    )
    
    args = parser.parse_args()
    
    # param
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')
    
    ### data preprocessing
    source = read_process_data_ARCHS4(args.source_data_file, args.source_label_file)
    data_source, source_labels = source[:,1:], source[:,0]
    #target = read_process_data_TCGA(args.target_data_file,args.target_label_file, replace=True)
    target = read_process_data_gtex(args.target_data_file,args.target_label_file)
    data_target, target_labels = target[:,1:], target[:,0]
    #test = read_process_data_TCGA(args.test_data_file,args.test_label_file, replace=True)
    test = read_process_data_gtex(args.test_data_file,args.test_label_file)
    x_test, y_test = test[:,1:], test[:,0]
    
    print(x_test.shape)
    print(y_test.shape)
    
    # proportion loop list
    prop_list = list(np.arange(0.001, 0.01, 0.001))
    prop_list += list(np.arange(0.01, 0.21, 0.01))
    
    # logs DataFrame
    df_iter = pd.DataFrame(columns=['pretrain_prop','pretrain_samples','iter','epoch','train_source_loss','valid_source_loss','test_loss','train_source_acc','valid_source_acc','test_acc'])
    df_prop = pd.DataFrame(columns=['pretrain_prop','pretrain_samples','iter','epoch','train_source_loss','valid_source_loss','test_loss','train_source_acc','valid_source_acc','test_acc'])
    df_final = pd.DataFrame(columns=['pretrain_prop','pretrain_samples','iter','epoch','train_source_loss','valid_source_loss','test_loss','train_source_acc','valid_source_acc','test_acc'])
    
    for prop in prop_list:
        df_prop = pd.DataFrame(columns=['pretrain_prop','pretrain_samples','iter','epoch','train_source_loss','valid_source_loss','test_loss','train_source_acc','valid_source_acc','test_acc'])
        for i in range(3):
            # split with proportion 
            train_idx, val_idx = generate_indices(data_target, target_labels, prop=0.01, val_prop=0.15, test_prop=0, rs=0)
            x_train_target, y_train_target = data_target[train_idx], target_labels[train_idx]
            x_valid_target, y_valid_target = data_target[val_idx], target_labels[val_idx]
            print(f'Target shape: {x_train_target.shape}')
    
            # split source
            pretrain_idx, pretrain_valid_idx = generate_indices(data_source, source_labels, prop=prop, val_prop=0.15, test_prop=0, rs=0, prog=True, safe_strat=True)
            x_train_source, x_valid_source = data_source[pretrain_idx], data_source[pretrain_valid_idx]
            y_train_source, y_valid_source = source_labels[pretrain_idx], source_labels[pretrain_valid_idx]
            print(x_train_target.shape, x_valid_target.shape, y_train_target.shape, y_valid_target.shape)
            print(x_train_source.shape, x_valid_source.shape, y_train_source.shape, y_valid_source.shape)
            print(np.unique(y_train_source), np.unique(y_valid_source))
            print(np.unique(y_train_target), np.unique(y_valid_target))
            print(np.unique(source_labels))
    
            # into datasets
            source_train_dataset = TensorDataset(torch.Tensor(x_train_source), torch.LongTensor(y_train_source))
            source_valid_dataset = TensorDataset(torch.Tensor(x_valid_source), torch.LongTensor(y_valid_source))
            target_train_dataset = TensorDataset(torch.Tensor(x_train_target), torch.LongTensor(y_train_target))
            target_valid_dataset = TensorDataset(torch.Tensor(x_valid_target), torch.LongTensor(y_valid_target))
            test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    
            # compute sampling weights
            num_source_train, num_target_train = len(x_train_source), len(x_train_target) 
            num_source_valid, num_target_valid = len(x_valid_source), len(x_valid_target)
            source_train_weights = [1.0/num_source_train] * num_source_train
            target_train_weights = [1.0/num_target_train] * num_target_train
            source_valid_weights = [1.0/num_source_valid] * num_source_valid
            target_valid_weights = [1.0/num_target_valid] * num_target_valid
            
            # create samplers 
            if num_source_train > num_target_train:
                source_train_sampler = WeightedRandomSampler(source_train_weights, num_samples=num_source_train, replacement=True)
                target_train_sampler = WeightedRandomSampler(target_train_weights, num_samples=num_source_train, replacement=True)  # Match source size
            else: 
                source_train_sampler = WeightedRandomSampler(source_train_weights, num_samples=num_target_train, replacement=True)
                target_train_sampler = WeightedRandomSampler(target_train_weights, num_samples=num_target_train, replacement=True)  # Match target size
            if num_source_valid > num_target_valid:
                source_valid_sampler = WeightedRandomSampler(source_valid_weights, num_samples=num_source_valid, replacement=True)
                target_valid_sampler = WeightedRandomSampler(target_valid_weights, num_samples=num_source_valid, replacement=True)  # Match source size
            else:
                source_valid_sampler = WeightedRandomSampler(source_valid_weights, num_samples=num_target_valid, replacement=True)
                target_valid_sampler = WeightedRandomSampler(target_valid_weights, num_samples=num_target_valid, replacement=True)  # Match target size

            # into DataLoaders
            source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, sampler=source_train_sampler, drop_last=True)
            source_valid_loader = DataLoader(source_valid_dataset, batch_size=args.batch_size, sampler=source_valid_sampler, drop_last=True)
            target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, sampler=target_train_sampler, drop_last=True)
            target_valid_loader = DataLoader(target_valid_dataset, batch_size=args.batch_size, sampler=target_valid_sampler, drop_last=True)

            source_train_loader_orig = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=False)
            source_valid_loader_orig = DataLoader(source_valid_dataset, batch_size=args.batch_size, shuffle=False)
            target_train_loader_orig = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=False)
            target_valid_loader_orig = DataLoader(target_valid_dataset, batch_size=args.batch_size, shuffle=False)
            
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
            # models parameters
            e_neuron_list = [256,256,256]
            c_neuron_list = [128, 64]
            d_neuron_list = [256,128,64]

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
            # early stopping
            early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='min')
    
            # logs lists
            train_source_acc_history, train_source_loss_history = [], []
            valid_source_acc_history, valid_source_loss_history = [], []
            valid_target_acc_history, valid_target_loss_history = [], []
            test_acc_history, test_loss_history = [], []
    
            start = time.time()
            for epoch in range(1, epochs+1):
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

                # eval on train source
                train_source_acc, train_source_loss = da_eval_epoch(model, source_train_loader_orig, f_loss_function, device, double_encoder=False)

                # eval on valid source/target
                valid_source_acc, valid_source_loss = da_eval_epoch(model, source_valid_loader_orig, f_loss_function, device, double_encoder=False)
                valid_target_acc, valid_target_loss = da_eval_epoch(model, target_valid_loader_orig, f_loss_function, device, double_encoder=False)
                     
                # eval on test
                test_acc, test_loss = da_eval_epoch(model, test_loader, f_loss_function, device, double_encoder=False)

                # update logs
                train_source_acc_history.append(train_source_acc)
                train_source_loss_history.append(train_source_loss)
                valid_source_acc_history.append(valid_source_acc)
                valid_source_loss_history.append(valid_source_loss)
                valid_target_acc_history.append(valid_target_acc)
                valid_target_loss_history.append(valid_target_loss)
                test_acc_history.append(test_acc)
                test_loss_history.append(test_loss)

                if early_stopping(valid_target_loss):
                    print(f"Early stopping triggered. Stopping training at epoch {epoch}")
                    print(f'Epoch {epoch}/{epochs} : loss = {valid_target_loss_history[-1]}')
                    break
            
            end = time.time()
            print(f'Training time: {end-start}')
    
            # save logs
            # df epochs
            df_iter = pd.DataFrame(columns=['pretrain_prop','pretrain_samples','iter','epoch','train_source_loss','valid_source_loss','test_loss','train_source_acc','valid_source_acc','test_acc'])
            df_iter['pretrain_prop'] = [prop for k in range(len(train_source_acc_history))]
            df_iter['iter'] = [i for k in range(len(train_source_acc_history))]
            df_iter['epoch'] = [k for k in range(len(train_source_acc_history))]
            df_iter['train_source_acc'] = train_source_acc_history
            df_iter['train_source_loss'] = train_source_loss_history
            df_iter['valid_source_acc'] = valid_source_acc_history
            df_iter['valid_source_loss'] = valid_source_loss_history
            df_iter['valid_target_acc'] = valid_target_acc_history
            df_iter['valid_target_loss'] = valid_target_loss_history
            df_iter['test_acc'] = test_acc_history
            df_iter['test_loss'] = test_loss_history
            df_iter['pretrain_samples'] = [len(x_train_source) for k in range(len(train_source_acc_history))]
            # merge iter to prop
            df_prop = pd.concat([df_prop, df_iter])
        # merge prop to final
        df_final = pd.concat([df_final, df_prop])
        df_final.to_csv(f'results/results_pretrain_proportion_{args.output_name}.csv')
    
if __name__ == "__main__":
    main()
        
        
    
    
