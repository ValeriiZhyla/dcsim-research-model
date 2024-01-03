# Start only one schedule-simulations.py, bad raise conditions otherwise

nohup python -u schedule-simulations.py --script_to_start=starters/start_third_phase_5.sh --directory_to_check=third_phase_5 --total_simulations_number=20 --minutes_between_checks=1 > scheduler_third_phase_5.log 2> scheduler_third_phase_5_error.log &


nohup python -u schedule-simulations.py --script_to_start=starters/start_third_phase_100.sh --directory_to_check=third_phase_100 --total_simulations_number=10 --minutes_between_checks=1 > scheduler_third_phase_100.log 2> scheduler_third_phase_100_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_third_phase_500.sh --directory_to_check=third_phase_500 --total_simulations_number=10 --minutes_between_checks=1 > scheduler_third_phase_500.log 2> scheduler_third_phase_500_error.log &


nohup python -u schedule-simulations.py --script_to_start=starters/start_third_phase_2000.sh --directory_to_check=third_phase_2000 --total_simulations_number=1000 --minutes_between_checks=1 > scheduler_third_phase_2000.log 2> scheduler_third_phase_2000_error.log &
