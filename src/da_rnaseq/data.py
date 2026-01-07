import random
import numpy as np 
import pandas as pd  
from sklearn import preprocessing  
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from math import ceil
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler, WeightedRandomSampler
import os
import torch
from sklearn.model_selection import KFold
from collections import defaultdict



# preprocessing functions
def read_process_data_TCGA(
    data_path,
    label_path,
    coding_genes=False,
    coding_genes_file='../../data/protein-coding_gene.txt',
    replace=True,
    return_df=False,
    binary=False
):
    class_df = pd.read_parquet(label_path)
    data_df = pd.read_parquet(data_path)
    
    if coding_genes:
        protein_coding_file = pd.read_csv(coding_genes_file, '\t')
        ens_list = np.unique(protein_coding_file['ensembl_gene_id'].tolist()).tolist()
        ens_list.pop()
        selected_columns = ['caseID'] + ens_list
        genes = data_df.columns
        intersection = list(set(selected_columns) & set(genes))
        data_df = data_df[intersection]

    # merging the dataframes based on "caseID"
    class_df["caseID"] = class_df.apply(lambda row: row.cases.split("|")[1], axis=1)
    df = class_df.merge(data_df, on="caseID", how="inner")
    
    if binary : df = df.drop(columns=list(df.columns[:6]) + [df.columns[7]] + [df.columns[8]] + [df.columns[9]])
    else: df = df.drop(columns=list(df.columns[:7]) + [df.columns[8]] + [df.columns[9]])  # columns management

    if replace and not binary:
        df = df.replace(['TCGA-BLCA','TCGA-BRCA','TCGA-CESC','TCGA-COAD','TCGA-LUAD','TCGA-LUSC','TCGA-OV','TCGA-PRAD','TCGA-SARC','TCGA-HNSC','TCGA-KIRC','TCGA-KIRP',
                        'TCGA-LGG','TCGA-LIHC','TCGA-SKCM','TCGA-STAD','TCGA-THCA','TCGA-UCEC'], 
                        ['bladder','breast','cervix','colon','lung','lung','ovary','prostate','sarcoma','headneck','kidney','kidney','brain','liver','skin','stomach','thyroid','uterine'])
    
    if return_df:
        return df
    
    else:
        # encoding cancer names to integers
        le = preprocessing.LabelEncoder()
        if binary: df["sample_type"] = le.fit_transform(df["sample_type"])
        else: df["cancer_type"] = le.fit_transform(df["cancer_type"])
        print(df.columns)

        
        np_dataset = df.to_numpy(dtype=np.float32)

        # normal standardardization
        scaler = preprocessing.StandardScaler()
        np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

        return np_dataset

def read_process_data_TCGA_unlabel(
    data_path,
    coding_genes=False,
    coding_genes_file='../../data/protein-coding_gene.txt',
    return_df=False
):
    data_df = pd.read_parquet(data_path)
    data_df = data_df.drop(columns="caseID")
    
    if coding_genes:
        protein_coding_file = pd.read_csv(coding_genes_file, '\t')
        selected_columns = np.unique(protein_coding_file['ensembl_gene_id'].tolist()).tolist()
        selected_columns.pop()
        genes = data_df.columns
        intersection = list(set(selected_columns) & set(genes))
        data_df = data_df[intersection]

    np_dataset = data_df.to_numpy(dtype=np.float32)

    # normal standardardization
    scaler = preprocessing.StandardScaler()
    np_dataset = scaler.fit_transform(np_dataset)

    return np_dataset


def read_process_data_gtex(
    data_path,
    label_path
):
    """Reads and processes (including a normal standardardization) the GTEx data.

    Args:
        data_path (str) : path to dataset
        label_path (str) : path to classes

    Returns:
        numpy.ndarray: Numpy array of the processed data.
    """
    
    class_df = pd.read_csv(label_path, sep='\t')
    data_df = pd.read_parquet(data_path)

    # merging the dataframes based on "caseID"
    class_df["cid"] = class_df['SAMPID']
    class_df = class_df.loc[:, class_df.columns.intersection(['cid','SMTS','SMTSD'])]  # columns management
    df = class_df.merge(data_df, on="cid", how="inner")
    
    df = df.drop(['cid','SMTSD'], axis=1)
    
    # replace
    df = df.replace(['Bladder', 'Brain', 'Breast', 'Cervix Uteri',
       'Colon', 'Esophagus','Salivary Gland','Kidney', 'Liver', 'Lung','Muscle','Adipose Tissue','Ovary',
       'Prostate', 'Skin', 'Stomach', 'Thyroid',
       'Uterus'], 
                        ['bladder','brain','breast','cervix','colon','headneck','headneck','kidney','liver','lung','sarcoma','sarcoma','ovary','prostate','skin','stomach','thyroid','uterine'])
    
    # encoding cancer names to integers
    le = preprocessing.LabelEncoder()
    df["SMTS"] = le.fit_transform(df["SMTS"])
    print(df.columns)
    np_dataset = df.to_numpy(dtype=np.float32)

    #normal standardardization
    scaler = preprocessing.StandardScaler()
    np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

    #return df
    return np_dataset


def read_process_data_ARCHS4(
    data_path,
    label_path,
    binary=False,
    coding_genes=False,
    coding_genes_file='../../data/protein-coding_gene.txt',
    return_df=False
):
    class_df = pd.read_parquet(label_path)
    data_df = pd.read_parquet(data_path)
    if binary : labels = class_df["cancer_type"]
    else : labels = class_df["labels"]
    
    if coding_genes:
        protein_coding_file = pd.read_csv(coding_genes_file, '\t')
        selected_columns = np.unique(protein_coding_file['ensembl_gene_id'].tolist()).tolist()
        selected_columns.pop()
        genes = data_df.columns
        intersection = list(set(selected_columns) & set(genes))
        data_df = data_df[intersection]
    
    if return_df:
        data_df.insert(0, 'cancer_type', class_df["labels"].tolist())
        return data_df

    else:
        # encoding cancer names to integers
        le = preprocessing.LabelEncoder()
        if binary : data_df.insert(0, 'cancer_type', le.fit_transform(class_df["cancer_type"]))
        else : data_df.insert(0, 'labels', le.fit_transform(class_df["labels"]))
        np_dataset = data_df.to_numpy(dtype=np.float32)

        # normal standardardization
        scaler = preprocessing.StandardScaler()
        np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

        return np_dataset

def read_process_data_ARCHS4_unlabel(
    data_path,
    binary=False,
    coding_genes=False,
    coding_genes_file='../../data/protein-coding_gene.txt'
):
    data_df = pd.read_parquet(data_path)
    
    if coding_genes:
        protein_coding_file = pd.read_csv(coding_genes_file, '\t')
        selected_columns = np.unique(protein_coding_file['ensembl_gene_id'].tolist()).tolist()
        selected_columns.pop()
        genes = data_df.columns
        intersection = list(set(selected_columns) & set(genes))
        data_df = data_df[intersection]
        
    np_dataset = data_df.to_numpy(dtype=np.float32)

    # normal standardardization
    scaler = preprocessing.StandardScaler()
    np_dataset = scaler.fit_transform(np_dataset)

    return np_dataset

"""def wrapper_tcga_archs4(
    tcga_data_path,
    tcga_label_path,
    archs4_data_path,
    archs4_label_path,
    archs4_prop =1
):
    tcga_dataset = read_process_data_TCGA(tcga_data_path, tcga_label_path)
    archs4_dataset = read_process_data_ARCHS4(archs4_data_path, archs4_label_path)
    if archs4_prop != 1:
        archs4_idx = generate_indices_pretraining(archs4_dataset, prop=archs4_prop)
        archs4_dataset = archs4_dataset[archs4_idx]
    
    dataset = np.concatenate((tcga_dataset, archs4_dataset))
    return dataset"""

def convert(data1, data2):
    # encoding cancer names to integers
    le = preprocessing.LabelEncoder()
    #data_df.insert(0, 'labels', le.fit_transform(class_df["labels"]))
    data1['cancer_type'] = le.fit_transform(data1['cancer_type'])
    data2['cancer_type'] = le.fit_transform(data2['cancer_type'])
    data_output1, data_output2 = data1, data2
    data_output1, data_output2 = data_output1.to_numpy(dtype=np.float32), data_output2.to_numpy(dtype=np.float32)

    # data1 normal standardardization
    scaler = preprocessing.StandardScaler()
    data_output1[:, 1:] = scaler.fit_transform(data_output1[:, 1:])

    # data2 normal standardardization
    scaler = preprocessing.StandardScaler()
    data_output2[:, 1:] = scaler.fit_transform(data_output2[:, 1:])
    
    return data_output1, data_output2


def read_process_data_artificial(data_path):
    df = pd.read_parquet(data_path)
    dataset = df.iloc[:, :-2]
    dataset = dataset.to_numpy(dtype=np.float32)
    
    labels = df.iloc[:,-1]
    labels = labels.to_numpy(dtype=np.float32)
    
    np_dataset = np.hstack((labels[:, None], dataset))
    
    # scaler
    scaler = preprocessing.StandardScaler()
    np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])
    
    return np_dataset


# Custom Dataset and DataLoader
class DA_Dataset(Dataset):
    def __init__(
        self,
        source_inputs,
        source_labels,
        target_inputs,
        device=None
    ):
        self.source_inputs = torch.from_numpy(source_inputs).to(dtype=torch.float32)
        self.source_labels = torch.from_numpy(source_labels).to(dtype=torch.long)
        self.target_inputs = torch.from_numpy(target_inputs).to(dtype=torch.float32)

        if device:
            self.source_inputs = self.source_inputs.to(device)
            self.source_labels = self.source_labels.to(device)
            self.target_inputs = self.target_inputs.to(device)

    def __len__(self):
        return len(self.source_inputs)

    def __getitem__(self, idx):
        source_label = self.source_labels[idx]
        source_values = self.source_inputs[idx]
        target_values = self.target_inputs[idx]
        return source_values, source_label, target_values
    
class DA_DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = bacth_size
        self.drop_last = drop_last
        
        sampler = RandomSampler(self.dataset)
        self.sampler = BatchSampler(sampler, batch_size, drop_last)
        
    def __iter__(self):
        self.idx_iterator = iter(self.sampler)
        return self
    
    def __next__(self):
        idx = next(self.idx_iterator)
        return self.dataset[idx]
    
    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        else:
            return ceil(length/self.batch_size)

    
def safe_stratified_train_val_split(indices, labels, val_prop=0.2, random_state=None):
    """
    Safely split indices into train and val sets with stratification,
    ensuring at least 1 sample per class in both sets.
    """
    indices = np.array(indices)
    labels = np.array(labels)
    
    # Build class -> list of index mappings
    class_to_indices = defaultdict(list)
    for idx, label in zip(indices, labels):
        class_to_indices[label].append(idx)
    
    train_idx = []
    val_idx = []
    
    rng = np.random.default_rng(random_state)
    
    for label, class_indices in class_to_indices.items():
        class_indices = np.array(class_indices)
        rng.shuffle(class_indices)
        n = len(class_indices)
        
        if n < 2:
            raise ValueError(f"Class '{label}' has only {n} sample(s), need at least 2 for stratified split.")
        
        n_val = max(1, int(round(n * val_prop)))
        n_train = n - n_val
        
        if n_train < 1:
            n_val -= 1
            n_train += 1
        
        val_idx.extend(class_indices[:n_val])
        train_idx.extend(class_indices[n_val:])
    
    return np.array(train_idx), np.array(val_idx)


def generate_indices(data, labels, prop=1, val_prop=0.15, test_prop=0, rs=0, prog=True, safe_strat=False):
    indices = list(range(len(data)))
    
    if test_prop != 0 :
        train_idx, test_idx = train_test_split(
            indices, test_size=test_prop, stratify=labels, train_size=None, random_state=rs
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_prop / (1 - test_prop),
            train_size=None,
            #stratify=data[train_idx,0],
            random_state=rs,
        )
    else :
        if not safe_strat:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_prop,
                train_size=None,
                stratify=labels,
                random_state=rs,
            )
        else:
            train_idx, val_idx = safe_stratified_train_val_split(indices, labels, val_prop=val_prop, random_state=rs)
        test_idx=[]
    if prop != 1:
        #modes = data[train_idx, 0]
        modes = labels[train_idx]
        #print(np.unique(modes))
        subtrain_idx = []
        for mode in np.unique(modes):
            candidates = np.array(train_idx)[np.argwhere(modes == mode).flatten()]
            # adding progressively 
            if prog==True: selected_idx = candidates[: round(np.ceil(len(candidates) * prop))]
            # random
            else: selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop) + 1), replace=False)
            #else: selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop)), replace=False)
            subtrain_idx += selected_idx.tolist()
        train_idx = subtrain_idx
        #print(np.unique(labels[subtrain_idx]))
        
    if test_prop == 0: return train_idx, val_idx
    else: return train_idx, val_idx, test_idx


def generate_indices_pretraining(data, labels, prop=1, rs=0):
    indices = list(range(len(data)))
    if prop != 1:
        modes = labels
        subtrain_idx = []
        for mode in np.unique(modes):
            candidates = np.array(indices)[np.argwhere(modes == mode).flatten()]
            # adding progressively 
            selected_idx = candidates[: round(len(candidates) * prop)]
            # random
            #selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop)), replace=False)
            subtrain_idx += selected_idx.tolist()
        train_idx = subtrain_idx
    return train_idx


def get_dataloaders(dataset, idx, bs=None, verbose=True, drop_last=True):
    if verbose:
        print(f"{len(dataset)} elements in the dataset")

    train_idx, val_idx, test_idx = idx

    if bs is None:
        bs = [max(1, len(train_idx)), max(1, len(val_idx)), max(1, len(test_idx))]

    train_sample = SubsetRandomSampler(train_idx)
    val_sample = SubsetRandomSampler(val_idx)
    test_sample = SubsetRandomSampler(test_idx)

    Dload = DataLoader

    trainset = Dload(dataset, batch_size=bs[0], sampler=train_sample, drop_last=drop_last)
    valset = Dload(dataset, batch_size=bs[1], sampler=val_sample, drop_last=drop_last)
    testset = Dload(dataset, batch_size=bs[2], sampler=test_sample, drop_last=drop_last)

    if verbose:
        print(f"{len(train_idx)} elements in the trainset")
        print(f"{len(val_idx)} elements in the valset")
        print(f"{len(test_idx)} elements in the testset")

    return (trainset, valset, testset)




class LogResults:
    """Class implementing an object for the monitoring of the training of a neural
    network. Hyperparameters of the training can be specified, abd metrics of each epoch
    of the training are logged. Logs can be saved at any time to a csv format file.

    Args:
        name (str): Name of the project.
        hyp_str (list of str): List of the names of hyperparameters.
        hyp_vals (list, optional): Values for the hyperparameters. Defaults to None.
    """

    def __init__(self, name, hyp_str, hyp_vals=None):
        self.name = name
        self.counter = 0
        self.df_results = pd.DataFrame(
            columns=["id", "epoch", "val_acc", "val_loss", "test_acc", "test_loss", "optim", "bn", "dropout_rate"]
            + hyp_str
        )
        self.hyp_vals = hyp_vals

    def log_epoch(
        self, epoch=None, valacc=None, valloss=None, testacc=None, testloss=None, optim=None, bn=None, dropout_rate=None
    ):
        """Logs last epoch metrics by appending it to a global Pandas dataframe.

        Args:
            epoch (int, optional): Epoch number. Defaults to None.
            valacc (float, optional): Validation accuracy. Defaults to None.
            valloss (float, optional): Validation loss. Defaults to None.
            testacc (float, optional): Test accuracy. Defaults to None.
            testloss (float, optional): Test loss. Defaults to None.
        """
        ###
        #print([f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate]+ self.hyp_vals)
        #print(self.df_results.columns)
        ###
        new_serie = [f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate] + self.hyp_vals
        """new_serie = pd.DataFrame(
            [f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate]
            + self.hyp_vals,
            index=self.df_results.columns,
        )"""
        
        self.df_results.loc[len(self.df_results)] = new_serie
        #self.df_results = self.df_results.append(new_serie, ignore_index=True)
        #self.df_results = pd.concat([self.df_results, new_serie])
        #print(self.df_results)

    def update_hyps(self, hyp_vals):
        """Changes the hyperparameters values.

        Args:
            hyp_vals (list): List of the new values for the hyperparameters.
        """
        self.hyp_vals = hyp_vals

    def next_run(self):
        """Stops monitoring the current run (training) and initiates a new one."""
        self.counter += 1

    def save_csv(self):
        """Saves the logs to a csv file. Saved to the current directory, the filed is named based on the project name."""
        self.df_results.to_csv(f"{self.name}.csv")

    def show_progression(self):
        """Prints last epochs of the training to the console."""
        print(f"\nIteration {self.counter}, below are the last 5 epochs :")
        print(self.df_results.tail(5))