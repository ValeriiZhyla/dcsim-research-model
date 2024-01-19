ID="16-kit_desy-cache-moved-to-desy-more-resources-in-desy-less-resources-in-kit-faster-links-extrapolation"
PATH_PREFIX="dcsim-configurations/4th-phase/extrapolation/${ID}"
PLATFORM="${PATH_PREFIX}/platform/WLCG_disklessTier2_reduced100-storage-cache-switched-more-resources-in-desy-less-resources-in-kit-faster-links.xml"
DATASET="${PATH_PREFIX}/dataset/realData_DESY_cache-giant_10000.json"
WORKLOAD_DIR="${PATH_PREFIX}/workload"
WORKLOAD_PREFIX="T2_DE_DESY_workloads_"
WORKLOAD_SUFFIX=".json"
BATCH_LENGTHS=(20000)

for LEN in "${BATCH_LENGTHS[@]}"; do
    WORKLOAD="${WORKLOAD_DIR}/${WORKLOAD_PREFIX}${LEN}${WORKLOAD_SUFFIX}"
    SIMULATION_DIR="${ID}/${LEN}"
    python start-simulations.py --simulations_number=50 --simulation_root_dir="$SIMULATION_DIR" --max_simulation_duration_hours=25 --platform_file="$PLATFORM" --dataset_file="$DATASET" --workload_file="$WORKLOAD"
done

