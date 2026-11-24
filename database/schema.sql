-- Run from Supabase SQL Editor. The Streamlit server uses a server-side
-- service-role key; never expose that key to client-side code or Git.
create extension if not exists pgcrypto;

create table if not exists public.medicines (
  medicine_id text primary key,
  medicine_name text not null check (length(trim(medicine_name)) > 0),
  category text not null check (length(trim(category)) > 0),
  current_stock double precision not null check (current_stock >= 0),
  average_daily_consumption double precision not null check (average_daily_consumption >= 0),
  supplier_lead_time double precision not null check (supplier_lead_time >= 0),
  safety_stock double precision not null check (safety_stock >= 0),
  reorder_quantity double precision not null check (reorder_quantity >= 0),
  manufacturing_date date not null,
  expiry_date date not null,
  supplier_name text not null check (length(trim(supplier_name)) > 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint valid_medicine_dates check (manufacturing_date <= expiry_date)
);

alter table public.medicines add column if not exists created_at timestamptz not null default now();
alter table public.medicines add column if not exists updated_at timestamptz not null default now();

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists medicines_set_updated_at on public.medicines;
create trigger medicines_set_updated_at before update on public.medicines
for each row execute function public.set_updated_at();

create table if not exists public.consumption_history (
  id bigint generated always as identity primary key,
  medicine_id text not null,
  medicine_name text not null,
  quantity double precision not null check (quantity > 0),
  consumed_on date not null,
  note text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists public.recommendation_history (
  id bigint generated always as identity primary key,
  provider text not null check (provider in ('gemini', 'rules')),
  recommendation text not null,
  inventory_snapshot jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists consumption_history_medicine_date_idx
  on public.consumption_history (medicine_id, consumed_on desc);
create index if not exists recommendation_history_created_at_idx
  on public.recommendation_history (created_at desc);

-- RLS blocks anon/authenticated access. Server-side service_role calls bypass
-- RLS; keep SUPABASE_KEY only in Streamlit server secrets/environment.
alter table public.medicines enable row level security;
alter table public.consumption_history enable row level security;
alter table public.recommendation_history enable row level security;
grant select, insert, update, delete on public.medicines to service_role;
grant select, insert, update, delete on public.consumption_history to service_role;
grant select, insert, update, delete on public.recommendation_history to service_role;
grant usage, select on all sequences in schema public to service_role;

create or replace function public.record_medicine_consumption(
  p_medicine_id text,
  p_medicine_name text,
  p_quantity double precision,
  p_consumed_on date,
  p_note text default ''
) returns void language plpgsql security definer set search_path = '' as $$
begin
  if p_quantity <= 0 then
    raise exception 'Consumption quantity must be greater than zero';
  end if;
  update public.medicines
     set current_stock = current_stock - p_quantity
   where medicine_id = p_medicine_id and current_stock >= p_quantity;
  if not found then
    raise exception 'Medicine not found or consumption exceeds current stock';
  end if;
  insert into public.consumption_history (medicine_id, medicine_name, quantity, consumed_on, note)
  values (p_medicine_id, coalesce(p_medicine_name, ''), p_quantity, p_consumed_on, coalesce(p_note, ''));
end;
$$;

revoke all on function public.record_medicine_consumption(text, text, double precision, date, text) from public, anon, authenticated;
grant execute on function public.record_medicine_consumption(text, text, double precision, date, text) to service_role;
