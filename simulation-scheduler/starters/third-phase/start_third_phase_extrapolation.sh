python start-simulations.py --simulations_number=100 --max_simulation_duration_hours=49 --platform_file=platform/WLCG_disklessTier2_reduced100.xml --dataset_file=extrapolation/realData_GridKA_cache-giant_5000.json --workload_file=extrapolation/T2_DE_DESY_workloads_10000.json  --simulation_root_dir=third_phase_extrapolation_x5

python start-simulations.py --simulations_number=100 --max_simulation_duration_hours=49 --platform_file=platform/WLCG_disklessTier2_reduced100.xml --dataset_file=extrapolation/realData_GridKA_cache-giant_x2_10000.json --workload_file=extrapolation/T2_DE_DESY_workloads_20000.json  --simulation_root_dir=third_phase_extrapolation_x10

python start-simulations.py --simulations_number=100 --max_simulation_duration_hours=49 --platform_file=platform/WLCG_disklessTier2_reduced100.xml --dataset_file=extrapolation/realData_GridKA_cache-giant_x4_20000.json --workload_file=extrapolation/T2_DE_DESY_workloads_40000.json  --simulation_root_dir=third_phase_extrapolation_x20
