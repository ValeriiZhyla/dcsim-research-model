# dcsim-research-model

## Scripts
### Data Generation
[start-simulations.py](simulation-starter%2Fstart-simulations.py) is a primary tool for initiating simulations. 
It begins by gathering necessary parameters, including the simulation platform details, workload characteristics, and dataset configurations. 
The script automatically generates a unique directory for each simulation instance. 
This directory is distinctly named using a combination of a timestamp and a universally unique identifier (UUID). 
The tool then proceeds to compose a SLURM job description file. 
This file specifies the execution parameters for the simulation when submitted to the BWUniCluster's job scheduling system. 
Afterwards, the tool automatically submits the simulation job to the cluster, if free job slots are available. 
Each simulation is started with its own simulation seed.

[schedule-simulations.py](simulation-scheduler%2Fschedule-simulations.py) monitors the cluster's job queue and identifies when there are free job slots available. 
As soon as it detects available slots in the cluster, this tool allocates these slots for new simulation jobs by executing start-simulations.py. 
The tool is designed to run autonomously, managing the complete cycle of filling up all available job slots, waiting for slots to become free, and then refilling them. It maintains a count of the simulations completed and terminates its operation once the predetermined number of simulations has been executed.
This tool is designed to be compatible with nohup (no hang-up), which allows users to start the tool and log out of the system, with the assurance that the simulations will continue running uninterrupted.



### Dataset Preparation
[extract-simulations.py](simulation-results-etl%2Fextract-simulations.py) reads the directories created by the start-simulations.py script. 
Each directory contains the input and output files of individual DCSim simulations and simulation execution metadata, such as CPU time and used memory. 
At first, this tool performs consistency checks on the simulation results. 
Upon successful validation, the tool proceeds to write the processed data into a designated database. 
Each simulation is identified by its UUID.

[prepare-dataset-commons.py](dataset_preparation%2Fprepare-dataset-commons.py) contains the common methods for export of the data from the database in CSV files, that will be used for model training and evaluation. 
This script also can enhance the dataset by adding extra features. 
Each simulation is grouped based on its unique UUID and ordered according to the job start time. 
It is used by scripts like [prepare-dataset-commons.py](dataset_preparation%2F3rd-phase%2Fprepare-dataset.py) in each use case.

### Model Training
TODO

## Pytorch
For pytorch installation with pip and CUDA support use for Windows 11
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
More information here: https://pytorch.org/get-started/locally/
CUDA Toolkit should be installed: https://developer.nvidia.com/cuda-zone