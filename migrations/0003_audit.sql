create table if not exists audit_events (
  id serial primary key,
  kind text not null,
  model_id integer,
  summary text not null,
  created_at timestamptz not null default now(),
  constraint audit_events_kind_chk check (kind in ('model_registered', 'review_recorded', 'batch_scored'))
);

create index if not exists audit_events_id_idx on audit_events (id desc);
