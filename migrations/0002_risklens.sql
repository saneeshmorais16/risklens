create table if not exists model_versions (
  id serial primary key,
  version_label text not null,
  contract_version text not null,
  contamination double precision not null,
  n_estimators integer not null,
  seed integer not null,
  train_days text not null,
  test_days text not null,
  n_train integer not null,
  n_test integer not null,
  n_test_injected integer not null,
  precision_at_flag double precision,
  recall_at_flag double precision,
  f1 double precision,
  roc_auc double precision,
  test_flag_rate double precision not null,
  threshold double precision not null,
  amount_p95 double precision not null,
  status text not null,
  artifact text not null,
  created_at timestamptz not null default now(),
  constraint model_versions_status_chk check (status in ('active', 'archived', 'candidate'))
);

create index if not exists model_versions_status_idx on model_versions (status);

create table if not exists scored_transactions (
  id serial primary key,
  model_version_id integer not null references model_versions (id),
  transaction_id text not null,
  customer_id text not null,
  amount double precision not null,
  hour integer not null,
  velocity_24h integer not null,
  distance_km double precision not null,
  new_device integer not null,
  event_day integer not null,
  anomaly_score double precision not null,
  flagged boolean not null,
  review_reasons text not null,
  injected_scenario boolean not null,
  split text not null,
  constraint scored_transactions_split_chk check (split in ('train', 'test'))
);

create index if not exists scored_tx_model_idx
  on scored_transactions (model_version_id, anomaly_score desc);

create table if not exists reviews (
  id serial primary key,
  scored_transaction_id integer not null references scored_transactions (id),
  decision text not null,
  reason_code text not null,
  created_at timestamptz not null default now(),
  constraint reviews_decision_chk check (decision in ('clear', 'hold', 'escalate')),
  constraint reviews_reason_chk check (
    reason_code in ('amount', 'velocity', 'distance', 'device', 'overnight', 'pattern')
  )
);

create index if not exists reviews_scored_idx
  on reviews (scored_transaction_id, id desc);
