SELECT number_of_rows, COUNT(simulation_id) as files FROM (SELECT
    simulation_id,
    COUNT(*) AS number_of_rows
FROM
    simulated_jobs
GROUP BY
    simulation_id) rows_in_each_simulation GROUP BY number_of_rows;
