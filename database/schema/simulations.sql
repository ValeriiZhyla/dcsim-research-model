create table simulations
(
    id                   uuid      not null
        constraint simulations_pk
            primary key,
    seed                 integer   not null,
    platform_config      xml       not null,
    dataset_config       json      not null,
    workload_config      json      not null,
    platform_config_name text      not null,
    dataset_config_name  text      not null,
    workload_config_name text      not null,
    created_at           timestamp not null,
    simulations_result   text      not null
);

comment on column simulations.id is 'simulation id';

comment on column simulations.seed is 'random seed used for simulation';

comment on column simulations.platform_config is 'platform configuration used in simulation';

comment on column simulations.dataset_config is 'dataset configuration used in simulation';

comment on column simulations.workload_config is 'workload configuration used in simulation';

comment on column simulations.platform_config_name is 'Name of platform config file';

comment on column simulations.dataset_config_name is 'Name of dataset config file';

comment on column simulations.workload_config_name is 'Name of  workload config file';

comment on column simulations.created_at is 'Date and time of simulation start';

comment on column simulations.simulations_result is 'Full text of simulation result CSV file';