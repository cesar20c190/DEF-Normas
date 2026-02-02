import argparse
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
import re
from datetime import datetime, timezone

DIARIO_API_BASE = os.getenv(
    "DIARIO_API_BASE", "https://diario-api.defensoria.ba.def.br/diario/getList"
)
DEFAULT_SQLITE_PATH = os.path.join("data", "ai_cache.sqlite")


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DiarioSync/1.0)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_diario_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS diario_cache (
            dof_id INTEGER PRIMARY KEY,
            dof_dh_publicacao TEXT,
            dof_nu_sequencial INTEGER,
            dof_nu_ano INTEGER,
            dof_extraordinario TEXT,
            arq_id INTEGER,
            arq_nm TEXT,
            arq_tp TEXT,
            arq_vl_tamanho INTEGER,
            arq_texto TEXT,
            source_url TEXT,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_diario_cache_publicacao
        ON diario_cache (dof_dh_publicacao)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_diario_cache_ano
        ON diario_cache (dof_nu_ano)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_diario_cache_numero
        ON diario_cache (dof_nu_sequencial)
        """
    )
    conn.commit()


def map_diario_entry(item: dict) -> dict:
    return {
        "dof_id": item.get("DOF_ID") or item.get("dof_id"),
        "dof_dh_publicacao": item.get("DOF_DH_PUBLICACAO") or item.get("dof_dh_publicacao"),
        "dof_nu_sequencial": item.get("DOF_NU_SEQUENCIAL") or item.get("dof_nu_sequencial"),
        "dof_nu_ano": item.get("DOF_NU_ANO") or item.get("dof_nu_ano"),
        "dof_extraordinario": item.get("DOF_EXTRAORDINARIO") or item.get("dof_extraordinario"),
        "arq_id": item.get("ARQ_ID") or item.get("arq_id"),
        "arq_nm": item.get("ARQ_NM") or item.get("arq_nm"),
        "arq_tp": item.get("ARQ_TP") or item.get("arq_tp"),
        "arq_vl_tamanho": item.get("ARQ_VL_TAMANHO") or item.get("arq_vl_tamanho"),
        "arq_texto": item.get("ARQ_TEXTO") or item.get("arq_texto"),
        "source_url": item.get("source_url"),
    }


def upsert_diario_sqlite(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = int(time.time())
    values = []
    for row in rows:
        values.append(
            (
                row.get("dof_id"),
                row.get("dof_dh_publicacao"),
                row.get("dof_nu_sequencial"),
                row.get("dof_nu_ano"),
                row.get("dof_extraordinario"),
                row.get("arq_id"),
                row.get("arq_nm"),
                row.get("arq_tp"),
                row.get("arq_vl_tamanho"),
                row.get("arq_texto"),
                row.get("source_url"),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO diario_cache (
            dof_id,
            dof_dh_publicacao,
            dof_nu_sequencial,
            dof_nu_ano,
            dof_extraordinario,
            arq_id,
            arq_nm,
            arq_tp,
            arq_vl_tamanho,
            arq_texto,
            source_url,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dof_id) DO UPDATE SET
            dof_dh_publicacao = excluded.dof_dh_publicacao,
            dof_nu_sequencial = excluded.dof_nu_sequencial,
            dof_nu_ano = excluded.dof_nu_ano,
            dof_extraordinario = excluded.dof_extraordinario,
            arq_id = excluded.arq_id,
            arq_nm = excluded.arq_nm,
            arq_tp = excluded.arq_tp,
            arq_vl_tamanho = excluded.arq_vl_tamanho,
            arq_texto = excluded.arq_texto,
            source_url = excluded.source_url,
            updated_at = excluded.updated_at
        """,
        values,
    )
    conn.commit()
    return len(values)


def supabase_upsert(base_url: str, api_key: str, table: str, rows: list[dict], on_conflict: str):
    if not rows:
        return 0
    url = f"{base_url.rstrip('/')}/rest/v1/{table}?on_conflict={urllib.parse.quote(on_conflict)}"
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    payload = json.dumps(rows).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Supabase upsert failed ({e.code}): {body}") from e
    return len(rows)


def fetch_diario_pages(
    search: str,
    data_inicio: str,
    data_fim: str,
    ano: str,
    tipo: str,
    numero: str,
    limit: int,
    max_pages: int,
):
    collected = []
    page = 1
    while True:
        params = {
            "page": str(page),
            "limit": str(limit),
            "search": search or "",
            "dataInicio": data_inicio or "",
            "dataFim": data_fim or "",
            "ano": ano or "",
            "tipo": tipo or "0",
            "numero": numero or "",
        }
        url = f"{DIARIO_API_BASE}?{urllib.parse.urlencode(params)}"
        payload = fetch_json(url)
        rows = payload if isinstance(payload, list) else payload.get("data") or payload.get("rows") or payload.get("results") or []
        if not rows:
            break
        collected.extend(rows)
        if len(rows) < limit:
            break
        page += 1
        if max_pages > 0 and page > max_pages:
            break
    return collected


def normalize_date_value(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    if re.match(r"^\d{2}/\d{2}/\d{4}$", text):
        day, month, year = text.split("/")
        return f"{year}-{month}-{day}"
    return text


def normalize_updated_at(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    text = str(value).strip()
    return text or None


def fetch_normativas_rows(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT url, title, ementa, date, tipo, text, updated_at FROM ai_cache"
    )
    rows = []
    for row in cur.fetchall():
        payload = dict(row)
        payload["date"] = normalize_date_value(payload.get("date"))
        payload["updated_at"] = normalize_updated_at(payload.get("updated_at"))
        rows.append(payload)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sincroniza SQLite com Supabase e Diário Oficial.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--supabase-url", default=os.getenv("https://kaemcsfdvsqcupgmenqn.supabase.co", ""))
    parser.add_argument(
        "--supabase-key",
        default=os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("REDACTED_SUPABASE_KEY", ""),
    )
    parser.add_argument("--supabase-diario-table", default="diario_entries")
    parser.add_argument("--supabase-normativas-table", default="normativas_entries")
    parser.add_argument("--sync-diario", action="store_true")
    parser.add_argument("--sync-normativas", action="store_true")
    parser.add_argument("--no-supabase", action="store_true")
    parser.add_argument("--no-sqlite", action="store_true")
    parser.add_argument("--diario-search", default="")
    parser.add_argument("--diario-start", default="")
    parser.add_argument("--diario-end", default="")
    parser.add_argument("--diario-ano", default="")
    parser.add_argument("--diario-tipo", default="0")
    parser.add_argument("--diario-numero", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args()

    sync_diario = args.sync_diario or not args.sync_normativas
    sync_normativas = args.sync_normativas or not args.sync_diario

    use_supabase = not args.no_supabase and args.supabase_url and args.supabase_key
    use_sqlite = not args.no_sqlite

    conn = None
    if use_sqlite:
        conn = sqlite3.connect(args.sqlite_path)
        ensure_diario_table(conn)

    if sync_diario:
        diario_raw = fetch_diario_pages(
            search=args.diario_search,
            data_inicio=args.diario_start,
            data_fim=args.diario_end,
            ano=args.diario_ano,
            tipo=args.diario_tipo,
            numero=args.diario_numero,
            limit=args.limit,
            max_pages=args.max_pages,
        )
        diario_rows = [map_diario_entry(item) for item in diario_raw]
        if use_sqlite and conn:
            stored = upsert_diario_sqlite(conn, diario_rows)
            print(f"[sqlite] diario_cache upsert: {stored}")
        if use_supabase:
            total = 0
            for batch in chunked(diario_rows, 200):
                total += supabase_upsert(
                    args.supabase_url,
                    args.supabase_key,
                    args.supabase_diario_table,
                    batch,
                    "dof_id",
                )
            print(f"[supabase] diario_entries upsert: {total}")
        if not use_supabase:
            print("[supabase] skip (missing url/key or disabled)")

    if sync_normativas:
        if not conn:
            conn = sqlite3.connect(args.sqlite_path)
        normativas_rows = fetch_normativas_rows(conn)
        if use_supabase:
            total = 0
            for batch in chunked(normativas_rows, 200):
                total += supabase_upsert(
                    args.supabase_url,
                    args.supabase_key,
                    args.supabase_normativas_table,
                    batch,
                    "url",
                )
            print(f"[supabase] normativas_entries upsert: {total}")
        else:
            print("[supabase] skip (missing url/key or disabled)")

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
