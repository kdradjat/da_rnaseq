import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F

from data import *


class Classif_Head(torch.nn.Sequential):
    def __init__(self, input_dim, output_size, neuron_list=[128, 64], dropout=0.0):
        layers = []
        in_dim = input_dim
        for layer_size in neuron_list:
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = layer_size

        layers.extend([nn.Linear(in_dim, output_size), nn.Dropout(dropout)])

        super().__init__(*layers)

class MLP_builder(torch.nn.Sequential):
    def __init__(self, input_dim, neuron_list=[256, 256, 256], dropout=0.0):
        layers = []
        in_dim = input_dim
        for layer_size in neuron_list:
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.Dropout(dropout))
            in_dim = layer_size

        super().__init__(*layers)

class Baseline_Model(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        encoder_neuron_list=[256,256,256],
        head_neuron_list=[128,64],
        dropout=0.0
    ):
        super().__init__()
        
        self.encoder = MLP_builder(input_dim,neuron_list=encoder_neuron_list, dropout=dropout)
        self.classif_head = Classif_Head(input_dim=encoder_neuron_list[-1], output_size=nb_classes, neuron_list=head_neuron_list, dropout=dropout)

    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.classif_head(embeddings)

        return embeddings, outputs
    
class Baseline_Model_SCARF(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        features_low,
        features_high,
        encoder_neuron_list=[256,256,256],
        head_neuron_list=[128,64],
        dropout=0.0,
        corruption_rate=0.3
    ):
        super().__init__()
        
        self.encoder = MLP_builder(input_dim,neuron_list=encoder_neuron_list, dropout=dropout)
        self.classif_head = Classif_Head(input_dim=encoder_neuron_list[-1], output_size=nb_classes, neuron_list=head_neuron_list, dropout=dropout)

    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.classif_head(embeddings)

        return embeddings, outputs

"""
class Discriminator_builder(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.extend([nn.Linear(in_dim, 1), nn.Dropout(dropout)])

        super().__init__(*layers)
"""
        
class Discriminator_builder(torch.nn.Sequential):
    def __init__(self, input_dim, neuron_list=[256, 128, 64], dropout=0.0):
        layers = []
        in_dim = input_dim
        for i, layer_size in enumerate(neuron_list):
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            layers.append(nn.Dropout(dropout))
            in_dim = layer_size

        layers.extend([nn.Linear(in_dim, 1)])

        super().__init__(*layers)


class DomainAdaptation_Model(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        e_neuron_list,
        c_neuron_list,
        d_neuron_list,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = MLP_builder(input_dim,e_neuron_list,dropout=dropout)
        self.classif_head = Classif_Head(input_dim=e_neuron_list[-1], output_size=nb_classes, neuron_list=c_neuron_list,dropout=dropout)

        self.discriminator = Discriminator_builder(
            input_dim=e_neuron_list[-1],
            neuron_list=d_neuron_list,
            dropout=dropout
        )
        
    def forward_f(self, x):
        embeddings = self.encoder(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs
    
    def forward_d(self, x):
        d_outputs = self.discriminator(x)
        return d_outputs
    
    def forward(self, x):
        embeddings = self.encoder(x)
        f_outputs = self.classif_head(embeddings)
        d_outputs = self.discriminator(embeddings)
        return embeddings, f_outputs, d_outputs
    

class DomainAdaptation_Model_2branches(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        e_neuron_list,
        c_neuron_list,
        d_neuron_list,
        dropout=0.0
    ):
        super().__init__()

        self.encoder_source = MLP_builder(input_dim,e_neuron_list,dropout=dropout)
        self.classif_head = Classif_Head(input_dim=e_neuron_list[-1], output_size=nb_classes, neuron_list=c_neuron_list,dropout=dropout)

        self.encoder_target = MLP_builder(input_dim,e_neuron_list,dropout=dropout)

        self.discriminator = Discriminator_builder(
            input_dim=e_neuron_list[-1],
            neuron_list=d_neuron_list,
            dropout=dropout
        )
        
    def forward_source(self, x):
        embeddings = self.encoder_source(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs

    def forward_target(self, x):
        embeddings = self.encoder_target(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs
    
    def forward_d(self, x):
        d_outputs = self.discriminator(x)
        return d_outputs
    
    def forward(self, x):
        embeddings = self.encoder(x)
        f_outputs = self.classif_head(embeddings)
        d_outputs = self.discriminator(embeddings)
        return embeddings, f_outputs, d_outputs


### DANN architecture ###
# Gradient Reversal Layer
from torch.autograd import Function
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Domain_Discriminator_builder(torch.nn.Sequential):
    def __init__(self, input_dim, neuron_list=[256, 128, 64], dropout=0.0):
        layers = []
        in_dim = input_dim
        for i, layer_size in enumerate(neuron_list):
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            layers.append(nn.Dropout(dropout))
            in_dim = layer_size

        layers.extend([nn.Linear(in_dim, 1)])
        layers.extend([nn.Sigmoid()])

        super().__init__(*layers)
        
    
class DANN_Model(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        e_neuron_list,
        c_neuron_list,
        d_neuron_list,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = MLP_builder(input_dim,e_neuron_list,dropout=dropout)
        self.classif_head = Classif_Head(input_dim=e_neuron_list[-1], output_size=nb_classes, neuron_list=c_neuron_list,dropout=dropout)

        self.discriminator = Domain_Discriminator_builder(
            input_dim=e_neuron_list[-1],
            neuron_list=d_neuron_list,
            dropout=dropout
        )
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
    def forward_f(self, x):
        embeddings = self.encoder(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs
    
    def forward_d(self, x):
        d_outputs = self.discriminator(x)
        return d_outputs
    
    def forward(self, x):
        embeddings = self.encoder(x)
        f_outputs = self.classif_head(embeddings)
        d_grl = self.grl(embeddings)
        d_outputs = self.discriminator(d_grl)
        return embeddings, f_outputs, d_outputs
    
    
class DANN_Model_2branches(torch.nn.Sequential):
    def __init__(
        self,
        input_dim,
        nb_classes,
        e_neuron_list,
        c_neuron_list,
        d_neuron_list,
        dropout=0.0
    ):
        super().__init__()

        self.encoder_source = MLP_builder(input_dim,e_neuron_list,dropout=dropout)
        self.classif_head = Classif_Head(input_dim=e_neuron_list[-1], output_size=nb_classes, neuron_list=c_neuron_list,dropout=dropout)

        self.encoder_target = MLP_builder(input_dim,e_neuron_list,dropout=dropout)

        self.discriminator = Domain_Discriminator_builder(
            input_dim=e_neuron_list[-1],
            neuron_list=d_neuron_list,
            dropout=dropout
        )
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
    def forward_source(self, x):
        embeddings = self.encoder_source(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs

    def forward_target(self, x):
        embeddings = self.encoder_target(x)
        outputs = self.classif_head(embeddings)
        return embeddings, outputs
    
    def forward_d(self, x):
        d_outputs = self.discriminator(x)
        return d_outputs
    
    def forward(self, x):
        embeddings = self.encoder(x)
        f_outputs = self.classif_head(embeddings)
        d_outputs = self.discriminator(embeddings)
        return embeddings, f_outputs, d_outputs
