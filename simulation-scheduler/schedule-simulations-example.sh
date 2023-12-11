# Start this tool with nohup and in background (with &) to prevent the termination on logout.

# use python -u for correct stdout redirection
nohup python -u schedule-simulations.py --script_to_start=starters/start_first_phase.sh --directory_to_check=first_phase --total_simulations_number=1000 --minutes_between_checks=5 &

# output of schedule-simulations.py and start-simulations.py is saved in nohup.out