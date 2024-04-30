# EDGE-Rec: Efficient and Data-Guided Edge Diffusion for Recommender System Graphs 
## *Authors: Edoardo Botta, Utkarsh Priyam, Hemit Shah*

This is the official repository of "EDGE-Rec: Efficient and Data-Guided Edge Diffusion for Recommender System Graphs", submitted as final project for the CMU course **[10-708: Probabilistic Graphical Models](https://andrejristeski.github.io/10708S24/)**.

### Training and Evaluation
We train and evaluate the model on the [ML-100k dataset](https://grouplens.org/datasets/movielens/100k/). We construct a custom 90-10 train-test split of the edges by adopting a stratified sampling approach to ensure that each user is represented in both the training and validation split.

We train on 1000 diffusion steps for 10000 iterations on a single A100 GPU in the Google Colab environment with batch size 16, patch size 50. 

### Replicability
Results can be replicated in a step-by-step fashion by running the [execute.ipynb](./execute.ipynb) notebook.
