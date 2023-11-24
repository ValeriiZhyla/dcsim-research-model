#!/bin/bash
#SBATCH --job-name=dcsim-simulation-uuid
#SBATCH --output=simulation-output/uuid-1284/job-output.txt
#SBATCH --error=simulation-output/uuid-1284/simulation-output.txt

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=single

#SBATCH --mail-user=jn1292@partner.kit.edu
#SBATCH --mail-type=FAIL

# Execute a command
./dc-sim --platform platform/WLCG_disklessTier2.xml --workload-configurations workload/T2_DE_DESY_workloads.json --dataset-configurations dataset/prefetchScan.json --output-file simulation-output/uuid-1284/jobs.csv --seed 424242 