module load compiler/gnu/12.1 

cd DCSim
build/dc-sim --platform data/platform-files/WLCG_disklessTier2.xml --workload-configurations data/workload-configs/T2_DE_DESY_workloads.json --dataset-configurations data/dataset-configs/prefetchScan.json --output-file example-output.csv --seed 424242 