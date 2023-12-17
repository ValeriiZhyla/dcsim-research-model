create table simulated_jobs_extrapolation
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
    constraint simulated_jobs_extrapolation_pk
        primary key (simulation_id, position_in_batch)
);


comment on column simulated_jobs_extrapolation.simulation_id is 'uuid of corresponding simulation. One simulation generates and processes one batch';

comment on column simulated_jobs_extrapolation.position_in_batch is 'job id in batch. Batch starts with 1';

comment on column simulated_jobs_extrapolation.tag is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.machine_name is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.hit_rate is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.job_start is 'part od DCSim output file. Seconds after simulation start';

comment on column simulated_jobs_extrapolation.job_end is 'part od DCSim output file. Seconds after simulation start';

comment on column simulated_jobs_extrapolation.compute_time is 'part of DCSim output file';

comment on column simulated_jobs_extrapolation.flops is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.input_files_transfer_time is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.input_files_size is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.output_files_transfer_time is 'part od DCSim output file';

comment on column simulated_jobs_extrapolation.output_files_size is 'part od DCSim output file';