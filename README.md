# dcsim-research-model

## Scripts
- [start-simulations.py](simulation-starter%2Fstart-simulations.py): generates and runs DCSim simulations as Slurm jobs. This tool will automate preparation and starting of each simulation.
- [schedule-simulations.py](simulation-scheduler%2Fschedule-simulations.py): starts simulation-starter.py, fills all available job slots, waits for free slots and fills them. Terminates when required number of simulations is completed. This tool will automate starting the needed number of simulations (and with nohup you can run it and logout)
- [extract-simulations.py](simulation-results-etl%2Fextract-simulations.py): reads the directories created by simulation-starter.py, processes the files and writes the content into database. This tool will move data from directories to database.
- [prepare-dataset.py](simulation-dataset-preparation%2Fprepare-dataset.py): exports the data from database in single CSV file, that will be used for model training and evaluating.

## Pytorch
For pytorch installation with pip and CUDA support use for Windows 11
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
More information here: https://pytorch.org/get-started/locally/
CUDA Toolkit should be installed: https://developer.nvidia.com/cuda-zone