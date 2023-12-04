create table simulated_jobs
(
    simulation_id              uuid             not null
        constraint jobs_simulations_id_fk
            references simulations,
    position_in_batch          integer          not null,
    tag                        text             not null,
    machine_name               text             not null,
    hit_rate                   double precision not null,
    job_start                  double precision not null,
    job_end                    double precision not null,
    compute_time               double precision not null,
    flops                      double precision not null,
    input_files_transfer_time  double precision not null,
    input_files_size           double precision not null,
    output_files_transfer_time double precision not null,
    output_files_size          double precision not null,
    constraint jobs_pk
        primary key (simulation_id, position_in_batch)
);

comment on column simulated_jobs.simulation_id is 'uuid of corresponding simulation. One simulation generates and processes one batch';

comment on column simulated_jobs.position_in_batch is 'job id in batch. Batch starts with 1';

comment on column simulated_jobs.tag is 'part od DCSim output file';

comment on column simulated_jobs.machine_name is 'part od DCSim output file';

comment on column simulated_jobs.hit_rate is 'part od DCSim output file';

comment on column simulated_jobs.job_start is 'part od DCSim output file. Seconds after simulation start';

comment on column simulated_jobs.job_end is 'part od DCSim output file. Seconds after simulation start';

comment on column simulated_jobs.compute_time is 'part of DCSim output file';

comment on column simulated_jobs.flops is 'part od DCSim output file';

comment on column simulated_jobs.input_files_transfer_time is 'part od DCSim output file';

comment on column simulated_jobs.input_files_size is 'part od DCSim output file';

comment on column simulated_jobs.output_files_transfer_time is 'part od DCSim output file';

comment on column simulated_jobs.output_files_size is 'part od DCSim output file';