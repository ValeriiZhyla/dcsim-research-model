ID="12-kit_desy-cache-moved-to-desy"
PATH_PREFIX="dcsim-configurations/4th-phase/${ID}"
PLATFORM="${PATH_PREFIX}/platform/WLCG_disklessTier2_reduced100-storage-is-cache-and-cache-is-storage.xml"
DATASET="${PATH_PREFIX}/dataset/realData_DESY_cache-large_2000.json"
WORKLOAD_DIR="${PATH_PREFIX}/workload"
WORKLOAD_PREFIX="T2_DE_DESY_workloads_"
WORKLOAD_SUFFIX=".json"
BATCH_LENGTHS=(5 10 20 50 100 250 500 1000 1500 2000)

for LEN in "${BATCH_LENGTHS[@]}"; do
    WORKLOAD="${WORKLOAD_DIR}/${WORKLOAD_PREFIX}${LEN}${WORKLOAD_SUFFIX}"
    SIMULATION_DIR="${ID}/${LEN}"
    python start-simulations.py --simulations_number=10 --simulation_root_dir="$SIMULATION_DIR" --max_simulation_duration_hours=25 --platform_file="$PLATFORM" --dataset_file="$DATASET" --workload_file="$WORKLOAD"
done

