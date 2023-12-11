ps -ef | grep 'schedule-simulations.py'

# If processes does not appear in ps, but you want to kill it - PID is always written in log just after start
cat scheduler_first_phase_1.log | grep 'PID'