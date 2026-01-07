# Domain Adaptation for Knowledge Transfer Across RNA-Seq Datasets
This repository contains the code used for the experiments in the article "Adversarial Domain Adaptation Enables Knowledge Transfer Acress RNA-Seq Datasets".<br>
The objective is to evaluate the application of Domain Adaptation on gene expression datasets through a deep learning-based framework. <br>

## Method
![](figures/archi_v2.png)

## Datasets
### The Cancer Genome Atlas (TCGA)
The Cancer Genome Atlas ([[TCGA]](https://portal.gdc.cancer.gov/)) collected many types of data for each of over 20,000 tumor and normal samples. Each step in the Genome Characterization Pipeline generated numerous data points, such as:
* clinical information (e.g., smoking status)
* molecular analyte metadata (e.g., sample portion weight)
* molecular characterization data (e.g., gene expression values)

### All RNA-Seq and ChIP-Seq Sample and Signature Search (ARCHS4)

### Genotype Tissue Expression (GTEx)

## Installation
```
git clone git@github.com:kdradjat/da_rnaseq.git
cd da_rnaseq
python3 -m venv venv
source venv/bin/activate
pip install .
```
## Usage 
All the scripts used for the experiments with the differents methods are available on the [scripts](https://github.com/kdradjat/da_rnaseq/tree/main/scripts) folder.
