create table simulation_metadata
(
    simulation_id uuid                     not null
        constraint simulation_metadata_pk
            primary key
        constraint simulation_metadata_simulations_id_fk
            references simulations,
    duration      interval                 not null,
    jobs          integer                  not null,
    started_at    timestamp with time zone not null,
    simulated_on  text                     not null
);

comment on column simulation_metadata.simulation_id is 'uuid of corresponding simulation';

comment on column simulation_metadata.duration is 'duration of simulation';

comment on column simulation_metadata.jobs is 'number of jobs processed in this simulation';

comment on column simulation_metadata.started_at is 'date and time of simulation start';

comment on column simulation_metadata.simulated_on is 'name or id of the server, where the simulation took place';


