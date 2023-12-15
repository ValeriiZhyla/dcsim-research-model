create view number_of_rows_in_each_simulation(simulation_id, number_of_rows) as
SELECT simulated_jobs.simulation_id,
       count(*) AS number_of_rows
FROM simulated_jobs
GROUP BY simulated_jobs.simulation_id;

create view number_of_simulations_by_row_count(number_of_rows, files) as
SELECT rows_in_each_simulation.number_of_rows,
       count(rows_in_each_simulation.simulation_id) AS files
FROM (SELECT simulated_jobs.simulation_id,
             count(*) AS number_of_rows
      FROM simulated_jobs
      GROUP BY simulated_jobs.simulation_id) rows_in_each_simulation
GROUP BY rows_in_each_simulation.number_of_rows;

create view get_100_simulations_per_group_by_row_number(simulation_id, number_of_rows) as
WITH rankedsimulations AS (SELECT number_of_rows_in_each_simulation.simulation_id,
                                  number_of_rows_in_each_simulation.number_of_rows,
                                  row_number()
                                  OVER (PARTITION BY number_of_rows_in_each_simulation.number_of_rows ORDER BY number_of_rows_in_each_simulation.simulation_id) AS rn
                           FROM number_of_rows_in_each_simulation)
SELECT rankedsimulations.simulation_id,
       rankedsimulations.number_of_rows
FROM rankedsimulations
WHERE rankedsimulations.rn <= 100;