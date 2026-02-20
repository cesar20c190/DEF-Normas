import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import re
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

DIARIO_API_BASE = os.getenv(
    "DIARIO_API_BASE", "https://diario-api.defensoria.ba.def.br/diario/getList"
)
DEFAULT_SQLITE_PATH = os.path.join("data", "ai_cache.sqlite")
DEFAULT_LEVANTAMENTO_DIR = "LEVANTAMENTO PORTARIAS"
SUPPORTED_LEVANTAMENTO_EXTENSIONS = {".rtf", ".odt", ".docx"}


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


def decode_bytes_with_fallback(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_text_from_xml(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    block_tags = {"p", "h", "list-item", "table-row", "line-break", "br"}
    space_tags = {"tab", "s"}
    parts: list[str] = []

    for element in root.iter():
        tag = element.tag.split("}", 1)[-1] if isinstance(element.tag, str) else ""
        if tag in block_tags and parts and not parts[-1].endswith("\n"):
            parts.append("\n")
        elif tag in space_tags:
            parts.append(" ")
        if element.text:
            parts.append(element.text)
        if element.tail:
            parts.append(element.tail)

    return normalize_text("".join(parts))


def extract_docx_text(file_path: str) -> str:
    try:
        with zipfile.ZipFile(file_path) as zipped:
            xml_bytes = zipped.read("word/document.xml")
    except (FileNotFoundError, KeyError, zipfile.BadZipFile):
        return ""
    return extract_text_from_xml(xml_bytes)


def extract_odt_text(file_path: str) -> str:
    try:
        with zipfile.ZipFile(file_path) as zipped:
            xml_bytes = zipped.read("content.xml")
    except (FileNotFoundError, KeyError, zipfile.BadZipFile):
        return ""
    return extract_text_from_xml(xml_bytes)


def extract_rtf_text(file_path: str) -> str:
    try:
        with open(file_path, "rb") as file:
            raw = file.read()
    except OSError:
        return ""

    text = decode_bytes_with_fallback(raw)
    output: list[str] = []
    stack: list[tuple[int, bool]] = []
    ignorable = False
    ucskip = 1
    curskip = 0
    index = 0

    while index < len(text):
        char = text[index]

        if char == "{":
            stack.append((ucskip, ignorable))
        elif char == "}":
            if stack:
                ucskip, ignorable = stack.pop()
            else:
                ucskip, ignorable = (1, False)
        elif char == "\\":
            index += 1
            if index >= len(text):
                break
            char = text[index]

            if char in ("\\", "{", "}"):
                if not ignorable and curskip == 0:
                    output.append(char)
            elif char == "'":
                hex_code = text[index + 1 : index + 3]
                if len(hex_code) == 2:
                    if not ignorable:
                        try:
                            output.append(bytes.fromhex(hex_code).decode("cp1252", errors="ignore"))
                        except ValueError:
                            pass
                    index += 2
            elif char == "*":
                ignorable = True
            else:
                match = re.match(r"([a-zA-Z]+)(-?\d+)? ?", text[index:])
                if match:
                    word = match.group(1)
                    arg = match.group(2)
                    index += len(match.group(0)) - 1

                    if word in {"par", "line"}:
                        if not ignorable:
                            output.append("\n")
                    elif word == "tab":
                        if not ignorable:
                            output.append("\t")
                    elif word == "uc":
                        if arg:
                            try:
                                ucskip = max(0, int(arg))
                            except ValueError:
                                pass
                    elif word == "u":
                        if arg and not ignorable:
                            try:
                                codepoint = int(arg)
                                if codepoint < 0:
                                    codepoint += 65536
                                output.append(chr(codepoint))
                            except ValueError:
                                pass
                        curskip = ucskip
                    elif word == "bin":
                        if arg:
                            try:
                                index += int(arg)
                            except ValueError:
                                pass
                    elif word in {
                        "fonttbl",
                        "colortbl",
                        "datastore",
                        "themedata",
                        "stylesheet",
                        "info",
                        "pict",
                        "object",
                        "xmlopen",
                        "xmlclose",
                    }:
                        ignorable = True
        elif char in "\r\n":
            pass
        else:
            if curskip > 0:
                curskip -= 1
            elif not ignorable:
                output.append(char)

        index += 1

    return normalize_text("".join(output))


def extract_supported_file_text(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".rtf":
        return extract_rtf_text(file_path)
    if extension == ".odt":
        return extract_odt_text(file_path)
    if extension == ".docx":
        return extract_docx_text(file_path)
    return ""


def normalize_title_from_filename(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    title = stem.replace("_", " ")
    title = re.sub(r"\s+", " ", title)
    return title.strip(" -")


def infer_tipo_from_title(title: str) -> str:
    upper = title.upper()
    if "RETIFICA" in upper:
        return "RETIFICACAO"
    if "REPUBLICA" in upper:
        return "REPUBLICACAO"
    if "PORTARIA" in upper:
        return "PORTARIA"
    return "NORMATIVA"


def infer_ementa_from_title(title: str) -> str | None:
    parts = re.split(r"\s+-\s+", title, maxsplit=1)
    if len(parts) == 2:
        ementa = parts[1].strip()
        return ementa or None
    return None


def infer_date_from_file(file_path: str, title: str) -> str | None:
    date_match = re.search(r"\b([0-3]?\d)[./-]([01]?\d)[./-]((?:19|20)\d{2})\b", title)
    if date_match:
        day = int(date_match.group(1))
        month = int(date_match.group(2))
        year = int(date_match.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d}"

    parent_folder = os.path.basename(os.path.dirname(file_path))
    if re.fullmatch(r"(?:19|20)\d{2}", parent_folder):
        return f"{parent_folder}-01-01"

    year_from_number = re.search(r"\b\d{1,4}[./-]((?:19|20)\d{2})\b", title)
    if year_from_number:
        return f"{year_from_number.group(1)}-01-01"

    year_match = re.search(r"\b((?:19|20)\d{2})\b", title)
    if year_match:
        return f"{year_match.group(1)}-01-01"

    return None


def build_local_normativa_url(root_dir: str, file_path: str) -> str:
    relative = os.path.relpath(file_path, root_dir).replace("\\", "/")
    return "local://levantamento-portarias/" + urllib.parse.quote(relative, safe="/")


def map_levantamento_file(root_dir: str, file_path: str, max_chars: int) -> dict | None:
    text = normalize_text(extract_supported_file_text(file_path))
    if not text:
        return None

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    file_name = os.path.basename(file_path)
    title = normalize_title_from_filename(file_name)
    updated_at = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).isoformat()

    return {
        "url": build_local_normativa_url(root_dir, file_path),
        "title": title,
        "ementa": infer_ementa_from_title(title),
        "date": infer_date_from_file(file_path, title),
        "tipo": infer_tipo_from_title(title),
        "text": text,
        "updated_at": updated_at,
    }


def fetch_levantamento_rows(
    root_dir: str,
    limit: int = 0,
    max_chars: int = 120000,
) -> tuple[list[dict], dict]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"DiretÃ³rio nÃ£o encontrado: {root_dir}")

    all_files: list[str] = []
    for current_root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            all_files.append(os.path.join(current_root, file_name))
    all_files.sort()

    rows: list[dict] = []
    stats = {
        "scanned": 0,
        "supported": 0,
        "imported": 0,
        "skipped_extension": 0,
        "skipped_empty": 0,
        "errors": 0,
    }

    for file_path in all_files:
        stats["scanned"] += 1
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in SUPPORTED_LEVANTAMENTO_EXTENSIONS:
            stats["skipped_extension"] += 1
            continue

        stats["supported"] += 1
        try:
            row = map_levantamento_file(root_dir, file_path, max_chars=max_chars)
        except Exception:
            stats["errors"] += 1
            continue

        if not row:
            stats["skipped_empty"] += 1
            continue

        rows.append(row)
        stats["imported"] += 1
        if limit > 0 and stats["imported"] >= limit:
            break

    return rows, stats


def dedupe_rows_by_url(rows: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for row in rows:
        row_url = str(row.get("url") or "").strip()
        if not row_url:
            continue
        deduped[row_url] = row
    return list(deduped.values())


def main():
    parser = argparse.ArgumentParser(description="Sincroniza SQLite com Supabase e Diario Oficial.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--levantamento-dir", default=DEFAULT_LEVANTAMENTO_DIR)
    parser.add_argument("--levantamento-limit", type=int, default=0)
    parser.add_argument("--levantamento-max-chars", type=int, default=120000)
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument(
        "--supabase-key",
        default=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY", ""),
    )
    parser.add_argument("--supabase-diario-table", default="diario_entries")
    parser.add_argument("--supabase-normativas-table", default="normativas_entries")
    parser.add_argument("--sync-diario", action="store_true")
    parser.add_argument("--sync-normativas", action="store_true")
    parser.add_argument("--sync-levantamento", action="store_true")
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

    selected_modes = args.sync_diario or args.sync_normativas or args.sync_levantamento
    if selected_modes:
        sync_diario = args.sync_diario
        sync_normativas = args.sync_normativas
        sync_levantamento = args.sync_levantamento
    else:
        # Legacy behavior: without mode flags, sync diary + normativas from SQLite.
        sync_diario = True
        sync_normativas = True
        sync_levantamento = False

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
        else:
            print("[supabase] skip diario (missing url/key or disabled)")

    normativas_rows: list[dict] = []

    if sync_normativas:
        if not conn:
            conn = sqlite3.connect(args.sqlite_path)
        sqlite_rows = fetch_normativas_rows(conn)
        normativas_rows.extend(sqlite_rows)
        print(f"[sqlite] normativas_rows loaded: {len(sqlite_rows)}")

    if sync_levantamento:
        levantamento_rows, levantamento_stats = fetch_levantamento_rows(
            root_dir=args.levantamento_dir,
            limit=max(0, int(args.levantamento_limit)),
            max_chars=max(1000, int(args.levantamento_max_chars)),
        )
        normativas_rows.extend(levantamento_rows)
        print(
            "[levantamento] "
            f"scanned={levantamento_stats['scanned']} "
            f"supported={levantamento_stats['supported']} "
            f"imported={levantamento_stats['imported']} "
            f"skipped_extension={levantamento_stats['skipped_extension']} "
            f"skipped_empty={levantamento_stats['skipped_empty']} "
            f"errors={levantamento_stats['errors']}"
        )

    if sync_normativas or sync_levantamento:
        normativas_rows = dedupe_rows_by_url(normativas_rows)
        print(f"[normativas] total upsert rows (dedupe by url): {len(normativas_rows)}")
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
            print("[supabase] skip normativas (missing url/key or disabled)")

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
