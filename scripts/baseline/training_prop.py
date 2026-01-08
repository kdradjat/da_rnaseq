import sys, os
import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from da_rnaseq.data import *
from da_rnaseq.models import *
from da_rnaseq.utils import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The training data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--train_label_file",
        default=None, 
        required=True,
        type=str,
        help='The training label file'
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        required=True,
        help="The test data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--test_label_file",
        default=None, 
        required=True,
        type=str,
        help='The test label file'
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int
    )
    parser.add_argument(
        "--nb_epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--output_name",
        default="model",
        type=str
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--binary",
        action='store_true',
        help='binary cancer/non-cancer classification'
    )
    
    args = parser.parse_args()
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')
    

    # data preprocessing - choose the adequate read_process_ function
    #source = read_process_data_ARCHS4(args.train_data_file, args.train_label_file, binary=args.binary)
    #target = read_process_data_ARCHS4(args.test_data_file, args.test_label_file, binary=args.binary)
    #source = read_process_data_TCGA(args.train_data_file, args.train_label_file, binary=args.binary)
    #target = read_process_data_TCGA(args.test_data_file, args.test_label_file, binary=args.binary)
    source = read_process_data_gtex(args.train_data_file, args.train_label_file)
    target = read_process_data_gtex(args.test_data_file, args.test_label_file)
    
    data_source, data_labels = source[:,1:], source[:,0]
    data_source = np.delete(data_source, -1, axis=1) # rm __indexlevel__
    x_test, y_test = target[:,1:], target[:,0]
    
    prop_list = (
        list(np.arange(0.01, 0.20, 0.01))
        + list(np.arange(0.20, 1.05, 0.05))
    
    )
    
    # logs DataFrame
    df_iter = pd.DataFrame(columns=['prop','iter','epoch','loss','valid_loss','acc','valid_acc','test_acc'])
    df_prop = pd.DataFrame(columns=['prop','iter','epoch','loss','valid_loss','acc','valid_acc','test_acc'])
    df_final = pd.DataFrame(columns=['prop','iter','epoch','loss','valid_loss','acc','valid_acc','test_acc'])
    
    for prop in prop_list:
        df_prop = pd.DataFrame(columns=['prop','iter','epoch','loss','valid_loss','acc','valid_acc','test_acc'])
        for i in range(3):
            # split with proportion
            train_idx, val_idx = generate_indices(data_source, data_labels, prop=prop, val_prop=0.15, test_prop=0, rs=0, prog=True, safe_strat=False)
            x_valid, y_valid = data_source[val_idx], data_labels[val_idx]
            print(f'Train shapes: {x_train.shape}, {y_train.shape}')
            print(f'Valid shapes: {x_valid.shape}, {y_valid.shape}')
            print(f'Test shapes: {x_test.shape}, {y_test.shape}')

            # param 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device)

            # into datasets/dataloaders
            train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
            valid_dataset = TensorDataset(torch.Tensor(x_valid), torch.LongTensor(y_valid))
            test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

            # build model
            model = Baseline_Model(
                input_dim=data_source.shape[1],
                nb_classes=len(np.unique(data_labels)),
                encoder_neuron_list=[256,256,256],
                head_neuron_list=[128,64],
                dropout=args.dropout
            ).to(device)

            # Training 
            epochs = args.nb_epochs
            optimizer = Adam(model.parameters(), lr=1e-4)
            loss_function = torch.nn.CrossEntropyLoss()

            loss_history = []
            valid_loss_history = []
            acc_history = []
            valid_acc_history = []
            test_acc_history = []
            
            early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')
            
            start = time.time()
            for epoch in range(epochs):
                # validation loss
                valid_loss = valid_loss_f(model, loss_function, valid_loader, device)
                valid_loss_history.append(valid_loss)
                # validation acc
                valid_acc = epoch_acc(model, valid_loader, device)
                valid_acc_history.append(valid_acc)
                # training + train loss
                train_loss = train_epoch(model, loss_function, train_loader, optimizer, device)
                loss_history.append(train_loss)
                # train acc
                train_acc = epoch_acc(model, train_loader, device)
                acc_history.append(train_acc)
                
                # test acc
                test_acc = epoch_acc(model, test_loader, device)
                test_acc_history.append(test_acc)
                
                # early stop
                if early_stopping(valid_loss):
                    print(f"Early stopping triggered. Stopping training at epoch {epoch}")
                    print(f'Epoch {epoch}/{epochs} : loss = {valid_loss}')
                    break

                
            end = time.time()
            print(f'Training time: {end-start}')

            # save history
            df_iter = pd.DataFrame(columns=['prop','iter','epoch','loss','valid_loss','acc','valid_acc','test_acc'])
            df_iter['prop'] = [prop for k in range(len(loss_history))]
            df_iter['iter'] = [i for k in range(len(loss_history))]
            df_iter['epoch'] = [k for k in range(len(loss_history))]
            df_iter['loss'] = loss_history
            df_iter['valid_loss'] = valid_loss_history
            df_iter['acc'] = acc_history
            df_iter['valid_acc'] = valid_acc_history
            df_iter['test_acc'] = test_acc_history
        
            # merge iter to prop
            df_prop = pd.concat([df_prop, df_iter])
        
        # merge prop to final
        df_final = pd.concat([df_final, df_prop])
        df_final.to_csv(f'results/results_proportion_{args.output_name}.csv')
            
    
    df_final.to_csv(f'results/results_proportion_{args.output_name}.csv')

if __name__ == "__main__" :
    main()