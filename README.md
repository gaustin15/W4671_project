# W4671_project

This repo has work for the W4671 final project. The goal is to explore how using DNABERT encodings on RNA reads can improve the representation of gut microbiome datasets. This is being explored through implementing hierarchal clustering of the otus based on the similarity of the encoded reads. All data used in the analysis can be found in the MicrobiomeHD resource (https://zenodo.org/record/569601#.X94xcC3Mzxs).

Setting up the Environment
-------------------------
Most of the environment setup follows the DNABERT instructions at https://github.com/jerryji1993/DNABERT. The full setup for this project is outlined below (note the relative paths for the cloned repos and created folders are important, so be sure to run the commands in the order outlined below):

Create Environment
```
conda create -n dnabert python=3.6
conda activate dnabert
```
Clone the repo, build folder to store data in, clone DNABERT and execute more installations
```
git clone https://github.com/gaustin15/W4671_project.git 
mkdir MicrobiomeHD_data
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
cd ../..
conda install -c bioconda sumaclust 
conda install seaborn
conda install -c anaconda ipykernel 
python -m ipykernel install --user --name=dnabert
```
To run each notebook, the corresponding dataset must de downloaded and unzipped in the `MicrobiomeHD/` folder. 


Outline of .py Files
----------------
| Python File | Description |
|--|--|
| `sbert_wk_functions.py` | Contains functions used to create the sbert-wk sentence embeddings. Most of this code is from their repo (https://github.com/BinWang28/SBERT-WK-Sentence-Embedding). The main function imported outside of this file is the `create_embeddings()` function, which takes as input a list of sentences (or in this case reads), a Bert formatted model, and a tokenizer. It returns 784-dimension vectorized representations of all input sentences/reads.|
| `test_models.py` | Has functions that are used to evaluate the predictions resulting from different clustering approaches. The `test_reduced_dims()` function evaluates DNABRT clusters, `test_reduce_by_phylo_name()` evaluates by phylogeny, and `test_reduced_sumaclust()` evalutates sumaclust clusters. All functions take as input an otu abundance data frame, a metadata data frame with a target variable to predict. Different functions also take as input different elements that are necessary for each clustering approach (similarity matrices, phylogenetic names, fasta file to run sumaclust on...). All functions run k-fold cross validation, and evalute the performance of the clustered model on a witheld test set. They return the roc plot values, the auc, and the number of clusters used the in modelling process. |

Outline of .ipynb Files
---------------------
All notebook files are running different clusters + predictions for different colorectal cancer datasets from MicrobiomeHD. As there are some slight differences across how data is stored from different studies, the preprocessing was kept within each notebook separately. All Notebooks are set up to run on a cpu, and do not require any additional computational power. One potential exception is the crc_large_dataset, which will run on a cpu, but it takes ~1 hour. All notebooks save plots to the `Results` folder, which are shown in the report.
| Notebook | Description |
|--|--|
| `CRC_zhao_prediction` | Runs clustering and predictions on MicrobiomeHD's crc_zhao... dataset (denoted as Dataset 1 in the report). Before running this file, download/unzip that file within a `MicrobiomeHD_data` folder that is outside the repo (see the relative paths in the notebooks for reference).  |
| `CRC_large_dataset_prediction` | Runs clustering and predictions on MicrobiomeHD's crc_baxter... dataset (denoted as Dataset 2 in the report). Before running this file, download/unzip that file within a `MicrobiomeHD_data` folder that is outside the repo (see the relative paths in the notebooks for reference). The notebook saves plots to the `Results` folder, which are shown in the report. |
| `CRC_xiang_prediction` | Runs clustering and predictions on MicrobiomeHD's crc_xiang... dataset (denoted as Dataset 3 in the report). Before running this file, download/unzip that file within a `MicrobiomeHD_data` folder that is outside the repo (see the relative paths in the notebooks for reference).  |



