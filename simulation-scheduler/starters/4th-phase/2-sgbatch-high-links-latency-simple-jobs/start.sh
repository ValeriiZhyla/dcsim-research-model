ID="2-sgbatch-high-links-latency-simple-jobs"
PATH_PREFIX="dcsim-configurations/4th-phase/${ID}"
PLATFORM="${PATH_PREFIX}/platform/sgbatch-high-latency-links.xml"
DATASET="${PATH_PREFIX}/dataset/uniform_RemoteStorage.json"
WORKLOAD_DIR="${PATH_PREFIX}/workload"
WORKLOAD_PREFIX="simple_uniform_"
WORKLOAD_SUFFIX=".json"
BATCH_LENGTHS=(1 10 20 50 100 250 500 1000 1500 2000)

for LEN in "${BATCH_LENGTHS[@]}"; do
    WORKLOAD="${WORKLOAD_DIR}/${WORKLOAD_PREFIX}${LEN}${WORKLOAD_SUFFIX}"
    SIMULATION_DIR="${ID}/${LEN}"
    python start-simulations.py --simulations_number=10 --simulation_root_dir="$SIMULATION_DIR" --max_simulation_duration_hours=25 --platform_file="$PLATFORM" --dataset_file="$DATASET" --workload_file="$WORKLOAD"
done

