-- Fix de performance para match_documents no Supabase/Postgres.
-- Objetivo: evitar statement timeout (~8s) em consultas vetoriais com alto volume.
--
-- Como usar:
-- 1) Abra o SQL Editor no Supabase.
-- 2) Rode este script inteiro.
-- 3) Teste novamente perguntar.js/server2.js.

create extension if not exists vector;

-- Indices de suporte ao filtro/join.
create index if not exists doc_embeddings_cliente_id_idx
  on public.doc_embeddings (cliente_id);

create index if not exists doc_embeddings_source_external_id_idx
  on public.doc_embeddings (source, external_id);

-- unified_documents pode ser view (nao indexavel) ou tabela/materialized view.
-- Este bloco so cria indice quando o objeto suporta indice.
do $$
declare
  v_relkind "char";
begin
  select c.relkind
    into v_relkind
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  where n.nspname = 'public'
    and c.relname = 'unified_documents'
  limit 1;

  if v_relkind in ('r', 'm') then
    execute 'create index if not exists unified_documents_source_external_id_idx on public.unified_documents (source, external_id)';
  else
    raise notice 'Pulando indice em public.unified_documents (relkind=%).', v_relkind;
  end if;
end
$$;

-- Indice vetorial (cosine) com lists=80 para caber em maintenance_work_mem=64MB.
-- O valor 80 evita o erro de memoria observado com lists=200.
create index if not exists doc_embeddings_embedding_ivfflat_idx
  on public.doc_embeddings
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 80);

analyze public.doc_embeddings;
do $$
declare
  v_relkind "char";
begin
  select c.relkind
    into v_relkind
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  where n.nspname = 'public'
    and c.relname = 'unified_documents'
  limit 1;

  if v_relkind in ('r', 'm', 'f') then
    execute 'analyze public.unified_documents';
  else
    raise notice 'Pulando ANALYZE em public.unified_documents (relkind=%).', v_relkind;
  end if;
end
$$;

-- Remove assinaturas antigas para evitar colisao de overload.
drop function if exists public.match_documents(vector, real, integer, text);
drop function if exists public.match_documents(vector, real, integer, text, text);
drop function if exists public.match_documents(vector, double precision, integer, text, text);
drop function if exists public.match_documents(vector(1536), real, integer, text);
drop function if exists public.match_documents(vector(1536), real, integer, text, text);
drop function if exists public.match_documents(vector(1536), double precision, integer, text, text);

create or replace function public.match_documents(
  query_embedding vector,
  match_threshold real default 0.4,
  match_count integer default 10,
  filtro_cliente_id text default null,
  query_text text default ''
)
returns table (
  content text,
  source text,
  external_id text,
  title text,
  similarity real
)
language plpgsql
stable
set search_path = public
as $$
declare
  v_threshold real := greatest(least(coalesce(match_threshold, 0.4), 1), 0);
  v_match_count integer := greatest(coalesce(match_count, 10), 1);
  v_probe_limit integer := least(greatest(v_match_count * 30, 200), 4000);
begin
  if query_embedding is null then
    return;
  end if;

  -- Ajuste de probes para equilibrar qualidade x latencia.
  perform set_config('ivfflat.probes', '10', true);

  return query
  with candidates as (
    select
      d.content,
      d.source,
      d.external_id::text as external_id,
      (1 - (d.embedding <=> query_embedding))::real as similarity
    from public.doc_embeddings d
    where d.embedding is not null
      and (
        filtro_cliente_id is null
        or filtro_cliente_id = ''
        or d.cliente_id = filtro_cliente_id
      )
    order by d.embedding <=> query_embedding
    limit v_probe_limit
  ),
  ranked as (
    select
      c.content,
      c.source,
      c.external_id,
      c.similarity,
      row_number() over (
        partition by c.source, c.external_id
        order by c.similarity desc
      ) as rn
    from candidates c
    where c.similarity >= v_threshold
  ),
  top_ranked as (
    select
      r.content,
      r.source,
      r.external_id,
      r.similarity
    from ranked r
    where r.rn = 1
    order by r.similarity desc
    limit v_match_count
  )
  select
    tr.content,
    tr.source,
    tr.external_id,
    u.title,
    tr.similarity
  from top_ranked tr
  left join public.unified_documents u
    on u.source = tr.source
   and u.external_id::text = tr.external_id
  order by tr.similarity desc;
end;
$$;

comment on function public.match_documents(vector, real, integer, text, text)
  is 'Busca vetorial otimizada (IVFFlat), dedup por documento e retorno com titulo.';

grant execute on function public.match_documents(vector, real, integer, text, text)
  to anon, authenticated, service_role;

-- Sanity check rapido (deve responder sem timeout).
-- with probe as (
--   select embedding from public.doc_embeddings where cliente_id = 'dpe-ba' limit 1
-- )
-- select *
-- from public.match_documents(
--   (select embedding from probe),
--   0.40,
--   5,
--   'dpe-ba',
--   ''
-- );
