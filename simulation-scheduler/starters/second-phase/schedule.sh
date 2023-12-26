# Start only one schedule-simulations.py, bad raise conditions otherwise

nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_10.sh --directory_to_check=second_phase_10 --total_simulations_number=1000 --minutes_between_checks=1 > scheduler_second_phase_10.log 2> scheduler_second_phase_10_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_20.sh --directory_to_check=second_phase_20 --total_simulations_number=1000 --minutes_between_checks=1 > scheduler_second_phase_20.log 2> scheduler_second_phase_20_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_50.sh --directory_to_check=second_phase_50 --total_simulations_number=1000 --minutes_between_checks=2 > scheduler_second_phase_50.log 2> scheduler_second_phase_50_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_100.sh --directory_to_check=second_phase_100 --total_simulations_number=1000 --minutes_between_checks=2 > scheduler_second_phase_100.log 2> scheduler_second_phase_100_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_250.sh --directory_to_check=second_phase_250 --total_simulations_number=1000 --minutes_between_checks=2 > scheduler_second_phase_250.log 2> scheduler_second_phase_250_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_500.sh --directory_to_check=second_phase_500 --total_simulations_number=1000 --minutes_between_checks=2 > scheduler_second_phase_500.log 2> scheduler_second_phase_500_error.log &
nohup python -u schedule-simulations.py --script_to_start=starters/start_second_phase_1000.sh --directory_to_check=second_phase_1000 --total_simulations_number=1000 --minutes_between_checks=2 > scheduler_second_phase_1000.log 2> scheduler_second_phase_1000_error.log &

