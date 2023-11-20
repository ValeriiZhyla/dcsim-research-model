create table jobs
(
    simulation_id              uuid             not null
        constraint jobs_simulations_id_fk
            references simulations,
    id                         int              not null,
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
        primary key (simulation_id, id)
);

comment on column jobs.simulation_id is 'uuid of corresponding simulation. One simulation generates and processes one batch';

comment on column jobs.id is 'job id in batch. Batch starts with 1';

comment on column jobs.tag is 'part od DCSim output file';

comment on column jobs.machine_name is 'part od DCSim output file';

comment on column jobs.hit_rate is 'part od DCSim output file';

comment on column jobs.job_start is 'part od DCSim output file. Seconds after simulation start';

comment on column jobs.job_end is 'part od DCSim output file. Seconds after simulation start';

comment on column jobs.compute_time is 'part of DCSim output file';

comment on column jobs.flops is 'part od DCSim output file';

comment on column jobs.input_files_transfer_time is 'part od DCSim output file';

comment on column jobs.input_files_size is 'part od DCSim output file';

comment on column jobs.output_files_transfer_time is 'part od DCSim output file';

comment on column jobs.output_files_size is 'part od DCSim output file';

