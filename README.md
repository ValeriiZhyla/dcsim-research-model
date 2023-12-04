# dcsim-research-model

## Scripts
- simulation-starter.py: generates and runs DCSim simulations as Slurm jobs. This tool will automate preparation and starting of each simulation.
- simulator-scheduler.py: starts simulation-starter.py, fills all available job slots, waits for free slots and fills them. Terminates when required number of simulations is completed. This tool will automate starting the needed number of simulations (and with nohup you can run it and logout)
- extract-simulations.py: reads the directories created by simulation-starter.py, processes the files and writes the content into database. This tool will move data from directories to database.