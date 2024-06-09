# Personalized Binomial DAGs

## Introduction

The repo is designed to provide the code and instructions to reproduce the experimental results and learn personalized Binomial DAGs on new data for the following paper:

> Personalized Binomial DAGs Learning with Network Structured Covariates

Please refer the paper for more details.

## Preparation
1. Please check the REQUIREMENTS.txt to install required packages for simulation and applications.

## Simulation - Linear Embedding
1. Before running the python command, customize the following parameters in dag_learnig_simu.py.  
random seed:
> seed = RANDOM SEED

save file:
> save_pth = YOUR PATH TO SAVE FILE

parameters:
> num_users = NUMER OF USERS IN THE SOCIAL NETWORK
> num_nodes = NUMER OF NODES IN THE DAG 

2. Run the command to estimate DAG over simulated data
> python dag_learning_simu.py


## Simulation - Nonlinear Embedding
1. Customize the following parameters in dag_learnig_simu.py, and set the parameter "nonlinear" to "True".  
random seed:
> seed = RANDOM SEED

save file:
> save_pth = YOUR PATH TO SAVE FILE

parameters:
> num_users = NUMER OF USERS IN THE SOCIAL NETWORK
> num_nodes = NUMER OF NODES IN THE DAG 
> nonlinear = True

2. Copy the file 'UserNet.npy' and 'UserFeat.npy' to the folder 'gae'.

3. Run the command for nonlinear embedding via GAEs.  
> python gae/train.py

4. Run the command to estimate DAG 
> python dag_learning_simu.py

## Real-world Application
1. Customize the following parameters in gae/train.py.  
usernet path:
> usernet_pth = PATH OF USER NETWORK

userfeat_pth:
> userfeat_pth = PATH OF USER FEAT

number of clusters:
> num_cluster = NUMBER OF CLUSTERS WITHIN THE SOCIAL NETWORK

2. Run the command for graph embedding by GAE.
> python gae/train.py

3. Customize the following parameters in dag_learning_real_data.py
total clicks:
> total_clicks = TOTAL NUMER OF CLICKS WITHIN UNIT TIME

number of clusters:
> n_clusters = NUMBER OF CLUSTERS WITHIN THE SOCIAL NETWORK

observation path:
> obs_pth = PATH OF OBSERVATION DATA TO ESTIMATE DAG

4. Run the command for DAG estimation:
> python dag_learning_real_data.py
