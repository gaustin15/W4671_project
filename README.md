# W4671_project

This repo has work for the W4671 final project. The goal is to explore how using DNABERT encodings on RNA reads can improve the representation of gut microbiome datasets. This is being explored through implementing hierarchal clustering of the otus based on the similarity of the encoded reads. All data used in the analysis can be found in the MicrobiomeHD resource (https://zenodo.org/record/569601#.X94xcC3Mzxs).


Outline of the Files
----------------

The `data` file contains multiple cohorts. In each, there are individual ASV files for every study, and a `metadata_merged.csv` file used for regression/classification.  
Description of the python files:
| Python File | Description |
|--|--|
| `sbert_wk_functions.py` | Contains functions used to create the sbert-wk sentence embeddings. Most of this code is from their repo (https://github.com/BinWang28/SBERT-WK-Sentence-Embedding). The main function imported outside of this file is the `create_embeddings()` function, which takes as input a list of sentences (or in this case reads), a Bert formatted model, and a tokenizer. It returns 784-dimension vectorized representations of all input sentences/reads.|
| `test_models.py` | Has functions that are used to evaluate the predictions resulting from different clustering approaches. The `test_reduced_dims()` function evaluates DNABRT clusters, `test_reduce_by_phylo_name()` evaluates by phylogeny, and `test_reduced_sumaclust()` evalutates sumaclust clusters. All functions take as input an otu abundance data frame, a metadata data frame with a target variable to predict. Different functions also take as input different elements that are necessary for each clustering approach (similarity matrices, phylogenetic names, fasta file to run sumaclust on...). All functions run k-fold cross validation, and evalute the performance of the clustered model on a witheld test set. They return the roc plot values, the auc, and the number of clusters used the in modelling process. |



