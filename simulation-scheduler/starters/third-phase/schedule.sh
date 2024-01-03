# Start only one schedule-simulations.py, bad raise conditions otherwise

nohup python -u schedule-simulations.py --script_to_start=starters/start_third_phase_5.sh --directory_to_check=third_phase_5 --total_simulations_number=10 --minutes_between_checks=1 > scheduler_third_phase_5.log 2> scheduler_third_phase_5_error.log &
