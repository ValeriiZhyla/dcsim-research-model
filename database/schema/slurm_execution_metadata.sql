create table slurm_execution_metadata
(
    simulation_id         uuid not null
        constraint simulation_metadata_pk
            primary key
        constraint simulation_metadata_simulations_id_fk
            references simulations,
    cpu_time_text         text not null,
    memory_used_text      text not null,
    slurm_output          text not null,
    slurm_job_description text not null
);

comment on column slurm_execution_metadata.simulation_id is 'uuid of corresponding simulation';

comment on column slurm_execution_metadata.cpu_time_text is 'CPU Time used for simulation';

comment on column slurm_execution_metadata.memory_used_text is 'Memory used for simulation';

comment on column slurm_execution_metadata.slurm_output is 'Full text of slurm output file';

comment on column slurm_execution_metadata.slurm_job_description is 'Job description used as slurm input';


