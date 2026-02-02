import json
import os
import re
import sqlite3
import urllib.request
import urllib.error
import urllib.parse
import uuid
import math
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

LANGCHAIN_AVAILABLE = False
GROQ_AVAILABLE = False
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(BASE_DIR, "html")
DATA_DIR = os.path.join(BASE_DIR, "data")
SAIDA_API_DIR = os.path.join(BASE_DIR, "saida_api")
DB_PATH = os.path.join(DATA_DIR, "ai_cache.sqlite")
DOCUMENTS_JSON_PATH = os.path.join(DATA_DIR, "documents.json")
IMAP_ENDPOINTS = {
    "api_Leis": "https://api.imap.org.br/api/Leis",
    "api_AtosAdministrativos": "https://api.imap.org.br/api/AtosAdministrativos",
}
DIARIO_API_BASE = os.getenv(
    "DIARIO_API_BASE", "https://diario-api.defensoria.ba.def.br/diario/getList"
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(DATA_DIR, "chroma_db"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "2000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "400"))
RAG_MAX_CHUNKS_PER_DOC = int(os.getenv("RAG_MAX_CHUNKS_PER_DOC", "40"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))
RAG_INDEX_BATCH = int(os.getenv("RAG_INDEX_BATCH", "120"))
DEBUG_LOGS = os.getenv("DEBUG_LOGS", "1") == "1"
STOPWORDS_PT = {
    "a",
    "o",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "por",
    "sobre",
    "que",
    "e",
    "ou",
    "se",
    "tem",
    "há",
    "existe",
    "existem",
    "alguma",
    "algum",
    "algumas",
    "alguns",
    "trata",
    "trate",
    "tratem",
    "norma",
    "normas",
    "lei",
    "leis",
    "ato",
    "atos",
    "administrativo",
    "administrativos",
}

os.makedirs(DATA_DIR, exist_ok=True)


def debug_log(message: str):
    if DEBUG_LOGS:
        print(message, flush=True)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_cache (
                url TEXT PRIMARY KEY,
                title TEXT,
                text TEXT NOT NULL,
                ementa TEXT,
                date TEXT,
                tipo TEXT,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_chat_history (
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_vector_index (
                url TEXT PRIMARY KEY,
                chunk_count INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
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
            CREATE INDEX IF NOT EXISTS idx_ai_chunks_url
            ON ai_chunks (url)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_chat_history_session
            ON ai_chat_history (session_id, created_at)
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ai_cache)")}
        for column_name, column_type in (
            ("ementa", "TEXT"),
            ("date", "TEXT"),
            ("tipo", "TEXT"),
        ):
            if column_name not in columns:
                conn.execute(f"ALTER TABLE ai_cache ADD COLUMN {column_name} {column_type}")
        conn.commit()


def sqlite_regexp(pattern: str, value: str) -> int:
    if value is None:
        return 0
    try:
        return 1 if re.search(pattern, value, re.IGNORECASE) else 0
    except re.error:
        return 0


def open_db():
    conn = sqlite3.connect(DB_PATH)
    conn.create_function("REGEXP", 2, sqlite_regexp)
    return conn


def get_cached_doc(url: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT url, title, text, ementa, date, tipo, updated_at FROM ai_cache WHERE url = ?",
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)


def save_cached_doc(payload: dict):
    url = payload.get("url")
    text = payload.get("text")
    title = payload.get("title")
    ementa = payload.get("ementa")
    date = payload.get("date")
    tipo = payload.get("tipo")
    if not url or not text:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO ai_cache (url, title, text, ementa, date, tipo, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                text = excluded.text,
                ementa = excluded.ementa,
                date = excluded.date,
                tipo = excluded.tipo,
                updated_at = strftime('%s', 'now')
            """,
            (url, title, text, ementa, date, tipo),
        )
        conn.commit()
    export_documents_to_json()
    return True


def export_documents_to_json():
    """Exporta todos os documentos do cache para um arquivo JSON para uso da IA"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT url, title, text, ementa, date, tipo FROM ai_cache ORDER BY updated_at DESC"
            )
            documents = [
                {
                    "id": row["url"],
                    "title": row["title"],
                    "text": row["text"],
                    "ementa": row["ementa"],
                    "date": row["date"],
                    "tipo": row["tipo"]
                }
                for row in cur.fetchall()
            ]
        with open(DOCUMENTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"documents": documents, "total": len(documents)}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Erro ao exportar documentos: {e}")
        return False


def normalize_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "results", "items"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return [payload]
    return []


def build_document_url(file_name: str) -> str:
    if not file_name:
        return ""
    return f"https://sai.io.org.br/Handler.ashx?f=f&query={urllib.parse.quote(str(file_name))}"


def fetch_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; CodexBot/1.0)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


_chroma_db: Optional["Chroma"] = None
_text_splitter: Optional["RecursiveCharacterTextSplitter"] = None


def get_groq_llm(temperature: float = 0.0):
    if not GROQ_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"select\s.+", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    sql = match.group(0).strip()
    sql = sql.split("\n\n")[0].strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def enforce_sql_safety(sql: str, limit: int = 10) -> str:
    if not sql:
        return ""
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        return ""
    if not re.search(r"\bai_cache\b", lowered):
        return ""
    if "limit" not in lowered:
        return f"{sql} LIMIT {limit}"
    return sql


def generate_sql_query(question: str, limit: int = 10) -> str:
    llm = get_groq_llm(temperature=0)
    if llm is None:
        debug_log(
            f"[GROQ SQL] LLM indisponível (GROQ_AVAILABLE={GROQ_AVAILABLE}, "
            f"GROQ_API_KEY=REDACTED
        )
        return ""
    debug_log(f"[GROQ SQL] Pergunta: {truncate_value(question, 200)} | limit={limit}")
    prompt = PromptTemplate.from_template(
        """Gere uma consulta SQL para SQLite usando SOMENTE a tabela ai_cache.

Schema da tabela ai_cache:
- url (TEXT)
- title (TEXT)
- text (TEXT)
- ementa (TEXT)
- date (TEXT)
- tipo (TEXT)
- updated_at (INTEGER)

Regras obrigatórias:
1) Use apenas SELECT.
2) Sempre inclua a coluna url no SELECT.
3) Use LIMIT {limit}.
4) Não use nenhuma outra tabela.
5) Para buscas por termos/assuntos, filtre nas colunas title, ementa e text.
6) Use a coluna date apenas se a pergunta citar data/ano explicitamente.
7) Evite filtrar somente por date quando a pergunta não menciona data.
8) Responda apenas com a SQL, sem explicações e sem markdown.

Pergunta do usuário: {question}

SQL:"""
    )
    response = llm.invoke(prompt.format(question=question, limit=limit))
    raw = response.content if hasattr(response, "content") else str(response)
    debug_log(f"[GROQ SQL] Resposta bruta: {truncate_value(raw, 1200)}")
    sql = extract_sql_from_text(raw)
    debug_log(f"[GROQ SQL] SQL extraída: {sql}")
    safe_sql = enforce_sql_safety(sql, limit=limit)
    if not safe_sql:
        debug_log("[GROQ SQL] SQL bloqueada pelo enforce_sql_safety")
    return safe_sql


def truncate_value(value, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def compact_rows_for_llm(
    rows: list,
    max_rows: int = 10,
    max_cell_chars: int = 1200,
    max_total_chars: int = 12000,
):
    compacted = []
    total = 0
    for row in rows[:max_rows]:
        new_row = []
        for value in row:
            cell = truncate_value(value, max_cell_chars)
            total += len(cell)
            if total > max_total_chars:
                return compacted
            new_row.append(cell)
        compacted.append(tuple(new_row))
    return compacted


def estimate_rows_chars(rows: list) -> int:
    total = 0
    for row in rows:
        for value in row:
            total += len(str(value or ""))
    return total


def execute_sql_query(sql: str):
    if not sql:
        debug_log("[SQL] Consulta vazia, nada para executar.")
        return [], []
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
    return rows, columns


def build_sql_answer(question: str, sql: str, rows: list, columns: list):
    llm = get_groq_llm(temperature=0)
    if llm is None:
        debug_log(
            f"[GROQ ANSWER] LLM indisponível (GROQ_AVAILABLE={GROQ_AVAILABLE}, "
            f"GROQ_API_KEY=REDACTED
        )
        return ""
    preview_rows = compact_rows_for_llm(rows, max_rows=10)
    debug_log(
        f"[GROQ ANSWER] rows={len(rows)} preview_rows={len(preview_rows)} "
        f"cols={len(columns)} preview_chars={estimate_rows_chars(preview_rows)}"
    )
    prompt = PromptTemplate.from_template(
        """Responda em português usando APENAS os dados das linhas abaixo.
Se os dados forem insuficientes ou pouco relevantes, explique o motivo de forma clara.

Como responder:
1) Resuma o que foi encontrado (títulos, tipos e datas relevantes).
2) Avalie a relevância para a pergunta (ex.: cita termos mas não regula o tema).
3) Dê a resposta. Se insuficiente, diga explicitamente e justifique.
4) Se couber, sugira termos alternativos de busca (ex.: jornada, expediente, turno).

Pergunta: {question}
SQL: {sql}
Colunas: {columns}
Linhas: {rows}

Resposta:"""
    )
    response = llm.invoke(
        prompt.format(question=question, sql=sql, columns=columns, rows=preview_rows)
    )
    return response.content if hasattr(response, "content") else str(response)


def get_text_splitter():
    global _text_splitter
    if _text_splitter is not None:
        return _text_splitter
    if not LANGCHAIN_AVAILABLE:
        return None
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _text_splitter


def get_chroma_db():
    global _chroma_db
    if _chroma_db is not None:
        return _chroma_db
    if not LANGCHAIN_AVAILABLE:
        return None
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    _chroma_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _chroma_db


def get_indexed_updated_at(url: str):
    if not url:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT updated_at FROM ai_vector_index WHERE url = ? LIMIT 1",
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return int(row[0] or 0)
        except (TypeError, ValueError):
            return 0


def mark_doc_indexed(url: str, chunk_count: int, updated_at: int | None = None):
    if not url:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO ai_vector_index (url, chunk_count, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                chunk_count = excluded.chunk_count,
                updated_at = excluded.updated_at
            """,
            (url, int(chunk_count), int(updated_at or 0)),
        )
        conn.commit()
    return True


def get_unindexed_docs(limit: int = 200):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT c.url, c.title, c.text, c.ementa, c.date, c.tipo, c.updated_at
            FROM ai_cache c
            LEFT JOIN ai_vector_index v ON c.url = v.url
            WHERE v.url IS NULL OR v.updated_at < c.updated_at
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def index_docs_in_chroma(docs: list[dict]) -> int:
    if not docs or not LANGCHAIN_AVAILABLE:
        return 0
    db = get_chroma_db()
    splitter = get_text_splitter()
    if db is None or splitter is None:
        return 0
    total_chunks = 0
    texts = []
    metadatas = []
    ids = []

    def flush():
        nonlocal total_chunks, texts, metadatas, ids
        if not texts:
            return
        db.add_texts(texts, metadatas=metadatas, ids=ids)
        total_chunks += len(texts)
        texts = []
        metadatas = []
        ids = []

    for doc in docs:
        url = doc.get("url")
        text = doc.get("text") or ""
        updated_at = int(doc.get("updated_at") or 0)
        if not url or not text:
            continue
        indexed_at = get_indexed_updated_at(url)
        if indexed_at is not None and indexed_at >= updated_at:
            continue
        if indexed_at is not None and indexed_at < updated_at:
            try:
                db.delete(where={"source": url})
            except Exception:
                pass
        chunks = splitter.split_text(text)
        if not chunks:
            continue
        if len(chunks) > RAG_MAX_CHUNKS_PER_DOC:
            chunks = chunks[:RAG_MAX_CHUNKS_PER_DOC]
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{url}#chunk_{idx}"
            texts.append(chunk)
            metadatas.append(
                {
                    "source": url,
                    "title": doc.get("title") or doc.get("ementa") or "Sem título",
                    "date": doc.get("date") or "",
                    "tipo": doc.get("tipo") or "",
                    "chunk_index": idx,
                }
            )
            ids.append(chunk_id)
            if len(texts) >= RAG_INDEX_BATCH:
                flush()
        mark_doc_indexed(url, len(chunks), updated_at)

    flush()
    return total_chunks


def retrieve_rag_chunks(query: str, top_k: int = RAG_TOP_K, score_threshold: float = RAG_SCORE_THRESHOLD):
    if not query or not LANGCHAIN_AVAILABLE:
        return []
    db = get_chroma_db()
    if db is None:
        return []
    results = db.similarity_search_with_relevance_scores(query, k=top_k)
    filtered = []
    for doc, score in results:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value >= score_threshold:
            filtered.append((doc, score_value))
    return filtered


def build_context_from_chunks(results: list[tuple], max_refs: int = 8):
    context = "TRECHOS MAIS RELEVANTES:\n\n"
    references = []
    seen_urls = set()
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        url = meta.get("source") or meta.get("url") or ""
        title = meta.get("title") or meta.get("ementa") or "Sem título"
        date = meta.get("date") or "Sem data"
        tipo = meta.get("tipo") or "Sem tipo"
        context += f"DOCUMENTO {i}:\n"
        context += f"- Título: {title}\n"
        context += f"- Data: {date}\n"
        context += f"- Tipo: {tipo}\n"
        if url:
            context += f"- URL: {url}\n"
        context += f"- Relevância: {score:.2f}\n"
        context += f"- Trecho relevante:\n{doc.page_content}\n"
        context += "\n" + "=" * 80 + "\n\n"

        if url and url not in seen_urls and len(references) < max_refs:
            seen_urls.add(url)
            references.append(
                {
                    "index": len(references) + 1,
                    "title": title,
                    "date": date,
                    "tipo": tipo,
                    "url": url,
                }
            )
    return context, references


def build_context_from_docs(docs: list[dict], search_mode: str, max_refs: int = 8):
    context = f"DOCUMENTOS ENCONTRADOS (busca {search_mode}):\n\n"
    references = []
    for i, doc in enumerate(docs, 1):
        url = doc.get("url") or ""
        title = doc.get("title") or doc.get("ementa") or "Sem título"
        date = doc.get("date") or "Sem data"
        tipo = doc.get("tipo") or "Sem tipo"
        context += f"DOCUMENTO {i}:\n"
        context += f"- Título: {title}\n"
        context += f"- Data: {date}\n"
        context += f"- Tipo: {tipo}\n"
        context += f"- Ementa: {doc.get('ementa', 'Sem ementa')}\n"
        context += f"- Conteúdo completo:\n{doc.get('text', '')}\n"
        context += "\n" + "=" * 80 + "\n\n"
        if url and len(references) < max_refs:
            references.append(
                {
                    "index": len(references) + 1,
                    "title": title,
                    "date": date,
                    "tipo": tipo,
                    "url": url,
                }
            )
    return context, references


def openai_post(path: str, payload: dict, api_key: str, timeout: int = 60):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    base = OPENAI_BASE_URL.rstrip("/")
    endpoint = path.lstrip("/")
    url = f"{base}/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API erro {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Falha ao conectar OpenAI: {e}") from e
    return json.loads(body)


def extract_openai_text(response: dict) -> str:
    if not isinstance(response, dict):
        return ""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = response.get("output", [])
    texts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text(text: str, max_chars: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + max_chars)
        chunks.append(cleaned[start:end].strip())
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(texts: list[str], task_type: str):
    if not texts:
        return []
    embeddings = []
    batch_size = 50
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return embeddings
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": OPENAI_EMBED_MODEL,
            "input": batch,
            "encoding_format": "float",
        }
        result = openai_post("embeddings", payload, api_key=api_key)
        data = result.get("data", [])
        data = sorted(data, key=lambda item: item.get("index", 0))
        for item in data:
            embeddings.append(item.get("embedding", []))
    return embeddings


def get_chunks_for_urls(urls: list[str], limit: int = 400):
    if not urls:
        return []
    rows = []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        step = 900
        for i in range(0, len(urls), step):
            subset = urls[i : i + step]
            placeholders = ",".join("?" for _ in subset)
            sql = f"""
                SELECT url, chunk_index, text, embedding
                FROM ai_chunks
                WHERE url IN ({placeholders})
                LIMIT ?
            """
            cur = conn.execute(sql, (*subset, limit))
            rows.extend(cur.fetchall())
    return [dict(row) for row in rows]


def has_chunks(url: str) -> bool:
    if not url:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT 1 FROM ai_chunks WHERE url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None


def store_chunks(url: str, chunks: list[str], embeddings: list[list[float]]):
    if not url or not chunks or not embeddings:
        return 0
    count = 0
    with sqlite3.connect(DB_PATH) as conn:
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                """
                INSERT INTO ai_chunks (url, chunk_index, text, embedding, created_at)
                VALUES (?, ?, ?, ?, strftime('%s', 'now'))
                """,
                (url, idx, chunk, json.dumps(emb)),
            )
            count += 1
        conn.commit()
    return count


def ensure_chunks_for_docs(docs: list[dict]):
    created = 0
    for doc in docs:
        url = doc.get("url") if isinstance(doc, dict) else None
        text = doc.get("text") if isinstance(doc, dict) else None
        if not url or not text:
            continue
        if has_chunks(url):
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        if len(chunks) > MAX_CHUNKS_PER_DOC:
            chunks = chunks[:MAX_CHUNKS_PER_DOC]
        embeddings = embed_texts(chunks, task_type="retrieval_document")
        if not embeddings or len(embeddings) != len(chunks):
            continue
        created += store_chunks(url, chunks, embeddings)
    return created


def get_doc_meta_by_urls(urls: list[str]):
    if not urls:
        return {}
    meta = {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        step = 900
        for i in range(0, len(urls), step):
            subset = urls[i : i + step]
            placeholders = ",".join("?" for _ in subset)
            sql = f"""
                SELECT url, title, ementa, date, tipo
                FROM ai_cache
                WHERE url IN ({placeholders})
            """
            cur = conn.execute(sql, subset)
            for row in cur.fetchall():
                meta[row["url"]] = dict(row)
    return meta


def semantic_search_chunks(query: str, urls: list[str], top_k: int = 8, per_doc_limit: int = 2):
    chunks = get_chunks_for_urls(urls, limit=600)
    if not chunks:
        return []
    query_embeds = embed_texts([query], task_type="retrieval_query")
    if not query_embeds:
        return []
    query_vec = query_embeds[0]
    scored = []
    for chunk in chunks:
        try:
            emb = json.loads(chunk.get("embedding") or "[]")
        except json.JSONDecodeError:
            emb = []
        score = cosine_similarity(query_vec, emb)
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    doc_counts = {}
    for item in scored:
        url = item.get("url")
        if not url:
            continue
        if doc_counts.get(url, 0) >= per_doc_limit:
            continue
        doc_counts[url] = doc_counts.get(url, 0) + 1
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def get_recent_docs(limit: int = 200):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT url, title, text, ementa, date, tipo
            FROM ai_cache
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def fetch_text_via_jina(url: str) -> str:
    cleaned = re.sub(r"^https?://", "", url)
    proxy_url = f"https://r.jina.ai/http://{cleaned}"
    req = urllib.request.Request(
        proxy_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CodexBot/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def resolve_imap_endpoints(names):
    if not names:
        return list(IMAP_ENDPOINTS.values())
    resolved = []
    for name in names:
        url = IMAP_ENDPOINTS.get(name)
        if url:
            resolved.append(url)
    return resolved


def fetch_and_cache_imap_docs(orgao: str, tipos: list[str], endpoints: list[str], limit: int = 30):
    stored = 0
    if not orgao or not tipos:
        return stored
    resolved = resolve_imap_endpoints(endpoints)
    if not resolved:
        return stored
    for endpoint in resolved:
        for tipo in tipos:
            if stored >= limit:
                return stored
            params = urllib.parse.urlencode(
                {
                    "cod_orgao_org": str(orgao),
                    "cod_tipo_ato_tia": str(tipo),
                }
            )
            url = f"{endpoint}?{params}"
            try:
                payload = fetch_json(url)
            except Exception:
                continue
            rows = normalize_payload(payload)
            for row in rows:
                if stored >= limit:
                    return stored
                file_name = row.get("des_nome_arquivo_mid") or row.get("des_nome_arquivo")
                if not file_name:
                    continue
                doc_url = build_document_url(file_name)
                if not doc_url:
                    continue
                if get_cached_doc(doc_url):
                    continue
                text = fetch_text_via_jina(doc_url)
                if not text:
                    continue
                title = (
                    row.get("des_ementa_lei")
                    or row.get("des_lei_lei")
                    or row.get("des_alt_arquivo_mid")
                    or row.get("des_ementa_ato")
                    or "Documento"
                )
                date = (
                    row.get("dat_publicacao_portal_lei")
                    or row.get("dat_publicacao_lei_lei")
                    or row.get("dat_publicacao")
                    or ""
                )
                tipo_label = (
                    row.get("des_tipo_ato_tia")
                    or row.get("des_legislacao_leg")
                    or row.get("des_sigla_tipo_ato_tia")
                    or ""
                )
                ementa = row.get("des_ementa_lei") or row.get("des_ementa_ato") or title
                ok = save_cached_doc(
                    {
                        "url": doc_url,
                        "title": str(title),
                        "text": text,
                        "ementa": str(ementa),
                        "date": str(date),
                        "tipo": str(tipo_label),
                    }
                )
                if ok:
                    stored += 1
    return stored


def save_chat_message(session_id: str, role: str, content: str):
    if not session_id or not role or not content:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO ai_chat_history (session_id, role, content, created_at)
            VALUES (?, ?, ?, strftime('%s', 'now'))
            """,
            (session_id, role, content),
        )
        conn.commit()
    prune_chat_history(session_id)
    return True


def load_chat_history(session_id: str, limit: int = 20):
    if not session_id:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT role, content, created_at
            FROM ai_chat_history
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cur.fetchall()
    rows = list(reversed(rows))
    return [dict(row) for row in rows]


def clear_chat_history(session_id: str):
    if not session_id:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM ai_chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
    return True


def prune_chat_history(session_id: str, keep: int = 40):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM ai_chat_history
            WHERE session_id = ?
              AND rowid NOT IN (
                SELECT rowid
                FROM ai_chat_history
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
              )
            """,
            (session_id, session_id, keep),
        )
        conn.commit()


def search_cached_docs(query: str, tipo: str | None, date_from: str | None, date_to: str | None, limit: int):
    query = (query or "").strip()
    exact = False
    if len(query) >= 2 and query.startswith('"') and query.endswith('"'):
        exact = True
        query = query[1:-1].strip()
    tokens = []
    if not exact and query:
        for token in re.findall(r"[\w-]+", query.lower()):
            if len(token) <= 2:
                continue
            if token in STOPWORDS_PT:
                continue
            tokens.append(token)
    with open_db() as conn:
        conn.row_factory = sqlite3.Row
        clauses = []
        params = []

        if query:
            if exact:
                clauses.append(
                    "(ementa REGEXP ? OR title REGEXP ? OR text REGEXP ?)"
                )
                pattern = rf"(^|\\W){re.escape(query)}(\\W|$)"
                params.extend([pattern, pattern, pattern])
            elif tokens:
                for token in tokens:
                    like = f"%{token}%"
                    clauses.append("(ementa LIKE ? OR title LIKE ? OR text LIKE ?)")
                    params.extend([like, like, like])
            else:
                like = f"%{query}%"
                clauses.append("(ementa LIKE ? OR title LIKE ? OR text LIKE ?)")
                params.extend([like, like, like])
        if tipo:
            clauses.append("tipo = ?")
            params.append(tipo)
        if date_from:
            clauses.append("date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("date <= ?")
            params.append(date_to)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT url, title, ementa, date, tipo, text, updated_at "
            "FROM ai_cache"
            f"{where} "
            "ORDER BY updated_at DESC "
            "LIMIT ?"
        )
        params.append(limit)
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?", 1)[0]
        path = path.split("#", 1)[0]
        rel = path.lstrip("/")
        
        # Se for requisição para /saida_api/, servir da pasta saida_api
        if rel.startswith("saida_api/"):
            return os.path.join(BASE_DIR, rel)
        
        # Caso contrário, servir da pasta html
        return os.path.join(HTML_DIR, rel)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/documents":
            try:
                with open(DOCUMENTS_JSON_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(data.encode("utf-8"))
            except FileNotFoundError:
                export_documents_to_json()
                try:
                    with open(DOCUMENTS_JSON_PATH, "r", encoding="utf-8") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(data.encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return
        if parsed.path == "/api/cache":
            query = parse_qs(parsed.query)
            url = query.get("url", [""])[0]
            if not url:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "url obrigatório"}).encode("utf-8"))
                return
            data = get_cached_doc(url)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            if data:
                payload = {"found": True, **data}
            else:
                payload = {"found": False}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
        if parsed.path == "/api/cache/search":
            query = parse_qs(parsed.query)
            q = query.get("q", [""])[0]
            tipo = query.get("tipo", [""])[0] or None
            date_from = query.get("date_from", [""])[0] or None
            date_to = query.get("date_to", [""])[0] or None
            try:
                limit = int(query.get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            limit = max(1, min(limit, 100))
            results = search_cached_docs(q, tipo, date_from, date_to, limit)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}).encode("utf-8"))
            return
        if parsed.path == "/api/ai/history":
            query = parse_qs(parsed.query)
            session_id = query.get("session_id", [""])[0]
            if not session_id:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "session_id obrigatório"}).encode("utf-8"))
                return
            try:
                limit = int(query.get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            limit = max(1, min(limit, 100))
            history = load_chat_history(session_id, limit)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"history": history}).encode("utf-8"))
            return
        if parsed.path == "/api/diario":
            query = parsed.query or ""
            url = f"{DIARIO_API_BASE}?{query}" if query else DIARIO_API_BASE
            try:
                payload = fetch_json(url)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Falha ao consultar diário: {e}"}).encode("utf-8"))
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/cache":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            if not isinstance(payload, dict):
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "json inválido"}).encode("utf-8"))
                return
            ok = save_cached_doc(payload)
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"saved": ok}).encode("utf-8"))
            return
        if parsed.path == "/api/ai":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            if not isinstance(payload, dict) or not payload.get("prompt"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "prompt obrigatório"}).encode("utf-8"))
                return

            session_id = payload.get("session_id") or uuid.uuid4().hex
            api_key = os.getenv("OPENAI_API_KEY")
            use_sql = bool(payload.get("use_sql", False))
            debug_log(
                "[AI] payload: "
                f"keys={list(payload.keys())} "
                f"use_sql={use_sql} "
                f"use_semantic={bool(payload.get('use_semantic', True))} "
                f"allow_fallback={bool(payload.get('allow_fallback', True))}"
            )
            debug_log(f"[AI] prompt={truncate_value(payload.get('prompt'), 200)}")
            if use_sql and not GROQ_AVAILABLE:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": "Dependências Groq/LangChain não instaladas. Instale langchain-groq."
                        }
                    ).encode("utf-8")
                )
                return
            if use_sql and not os.getenv("GROQ_API_KEY"):
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "GROQ_API_KEY não configurada"}).encode("utf-8"))
                return
            if not use_sql and not api_key:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "OPENAI_API_KEY não configurada"}).encode("utf-8"))
                return

            try:
                if use_sql:
                    user_prompt = payload["prompt"]
                    try:
                        sql_limit = int(payload.get("sql_limit", 8))
                    except (TypeError, ValueError):
                        sql_limit = 8
                    sql_limit = max(1, min(sql_limit, 50))
                    sql = generate_sql_query(user_prompt, limit=sql_limit)
                    if sql:
                        debug_log(f"[SQL] gerada: {sql}")
                    else:
                        debug_log("[SQL] SQL vazia gerada pelo Groq")
                    rows, columns = execute_sql_query(sql)
                    debug_log(f"[SQL] linhas={len(rows)} colunas={columns}")
                    answer = build_sql_answer(user_prompt, sql, rows, columns)
                    if not answer:
                        answer = "Não foi possível gerar análise a partir do banco de dados."

                    references = []
                    if rows and columns and "url" in columns:
                        url_idx = columns.index("url")
                        seen = set()
                        for row in rows:
                            if url_idx >= len(row):
                                continue
                            url = row[url_idx]
                            if not url or url in seen:
                                continue
                            seen.add(url)
                            references.append(
                                {
                                    "index": len(references) + 1,
                                    "title": "",
                                    "date": "",
                                    "tipo": "",
                                    "url": str(url),
                                }
                            )
                            if len(references) >= 8:
                                break
                    if references:
                        refs_lines = []
                        for ref in references:
                            refs_lines.append(ref["url"])
                        answer = f"{answer}\n\nReferências:\n" + "\n".join(refs_lines)

                    save_chat_message(session_id, "user", user_prompt)
                    save_chat_message(session_id, "assistant", answer)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "answer": answer,
                                "session_id": session_id,
                                "references": references,
                                "fallback_used": False,
                                "fallback_stored": 0,
                                "semantic_used": False,
                                "semantic_chunks": 0,
                            }
                        ).encode("utf-8")
                    )
                    return

                if not LANGCHAIN_AVAILABLE:
                    raise RuntimeError(
                        "Dependências LangChain/Chroma não instaladas. "
                        "Instale langchain, langchain-openai, langchain-chroma e chromadb."
                    )

                # Buscar documentos relevantes usando a pergunta do usuário
                user_prompt = payload["prompt"]
                history = load_chat_history(session_id, limit=12)
                use_semantic = bool(payload.get("use_semantic", True))
                fallback_used = False
                fallback_stored = 0

                doc_pattern = re.search(
                    r'(Portaria|Lei|Decreto|Resolução|Ato)\s*n?[°º]?\s*(\d+[\./\-]\d+)',
                    user_prompt,
                    re.IGNORECASE,
                )
                relevant_docs = []
                search_mode = "geral"
                if doc_pattern:
                    doc_identifier = doc_pattern.group(0)
                    relevant_docs = search_cached_docs(doc_identifier, None, None, None, 5)
                    if relevant_docs:
                        search_mode = "específica"

                # Indexar documentos pendentes no Chroma antes da busca vetorial
                if use_semantic:
                    pending_docs = get_unindexed_docs(limit=RAG_INDEX_BATCH)
                    if pending_docs:
                        index_docs_in_chroma(pending_docs)

                semantic_chunks = []
                documents_context = ""
                references = []

                if relevant_docs:
                    index_docs_in_chroma(relevant_docs)
                    documents_context, references = build_context_from_docs(relevant_docs, search_mode)
                elif use_semantic:
                    top_k = int(payload.get("semantic_top_k", RAG_TOP_K))
                    try:
                        score_threshold = float(payload.get("rag_score_threshold", RAG_SCORE_THRESHOLD))
                    except (TypeError, ValueError):
                        score_threshold = RAG_SCORE_THRESHOLD
                    semantic_chunks = retrieve_rag_chunks(
                        user_prompt,
                        top_k=top_k,
                        score_threshold=score_threshold,
                    )

                if not semantic_chunks and not documents_context:
                    allow_fallback = bool(payload.get("allow_fallback", True))
                    orgao = payload.get("orgao") or payload.get("cod_orgao_org") or ""
                    tipos = payload.get("tipos") or []
                    endpoints = payload.get("imap_endpoints") or []
                    if isinstance(tipos, str):
                        tipos = [tipos]
                    try:
                        fallback_limit = int(payload.get("fallback_limit", 30))
                    except ValueError:
                        fallback_limit = 30
                    fallback_limit = max(1, min(fallback_limit, 50))

                    if allow_fallback and orgao and tipos:
                        fallback_stored = fetch_and_cache_imap_docs(
                            orgao=str(orgao),
                            tipos=[str(tipo) for tipo in tipos],
                            endpoints=endpoints if isinstance(endpoints, list) else [],
                            limit=fallback_limit,
                        )
                        if fallback_stored > 0:
                            fallback_used = True
                            pending_docs = get_unindexed_docs(limit=RAG_INDEX_BATCH)
                            if pending_docs:
                                index_docs_in_chroma(pending_docs)
                            if use_semantic:
                                semantic_chunks = retrieve_rag_chunks(
                                    user_prompt,
                                    top_k=top_k,
                                    score_threshold=score_threshold,
                                )

                if semantic_chunks:
                    documents_context, references = build_context_from_chunks(semantic_chunks)
                elif not documents_context and not use_semantic:
                    relevant_docs = search_cached_docs(user_prompt, None, None, None, 20)
                    if relevant_docs:
                        documents_context, references = build_context_from_docs(relevant_docs, "geral")

                if not documents_context:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "answer": "Não encontrei documentos relevantes na base de dados para responder sua pergunta.",
                                "fallback_used": fallback_used,
                                "fallback_stored": fallback_stored,
                                "session_id": session_id,
                            }
                        ).encode("utf-8")
                    )
                    return

                # Ajustar prompt conforme tipo de busca
                if search_mode == "específica":
                    instruction = (
                        f"ATENÇÃO: O usuário perguntou especificamente sobre '{doc_pattern.group(0)}'. "
                        "Responda APENAS sobre este documento específico. "
                        "Se houver mais de um documento listado, foque no que corresponde exatamente ao solicitado."
                    )
                else:
                    instruction = "ATENÇÃO: Analise os documentos listados e responda com base nos relevantes."

                history_context = ""
                if history:
                    history_lines = []
                    for item in history:
                        role = "Usuário" if item.get("role") == "user" else "Assistente"
                        history_lines.append(f"{role}: {item.get('content', '')}")
                    history_context = "HISTÓRICO DA CONVERSA:\n" + "\n".join(history_lines) + "\n\n"

                full_prompt = f"""Você é um assistente de análise de documentos. Analise EXCLUSIVAMENTE os documentos listados abaixo.

{documents_context}

{history_context}PERGUNTA ATUAL: {user_prompt}

{instruction}

COMO RESPONDER:
1. Se a pergunta é sobre um documento específico: apresente APENAS as informações daquele documento
2. Cite o número do documento, título e data.
3. Use apenas informações presentes nos documentos listados
4. Se nenhum documento responder, diga: "Nenhum dos documentos encontrados responde a esta pergunta."

Use SOMENTE o conteúdo dos documentos listados acima."""

                llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2)
                answer = llm.invoke(full_prompt).content

                if references:
                    refs_lines = []
                    for ref in references:
                        url = ref.get("url") or ""
                        refs_lines.append(
                            f"[{ref['index']}] {ref['title']} — {ref['date']} — {ref['tipo']}"
                            + (f" — {url}" if url else "")
                        )
                    answer = f"{answer}\n\nReferências:\n" + "\n".join(refs_lines)

                save_chat_message(session_id, "user", user_prompt)
                save_chat_message(session_id, "assistant", answer)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "answer": answer,
                            "session_id": session_id,
                            "references": references,
                            "fallback_used": fallback_used,
                            "fallback_stored": fallback_stored,
                            "semantic_used": bool(semantic_chunks),
                            "semantic_chunks": len(semantic_chunks),
                        }
                    ).encode("utf-8")
                )

            except Exception as e:
                print(f"Erro ao chamar IA: {e}")
                self.send_response(502)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Falha ao chamar IA: {e}"}).encode("utf-8"))
            return
        super().do_POST()


if __name__ == "__main__":
    init_db()
    export_documents_to_json()
    server = HTTPServer(("", 8000), Handler)
    print("Servidor iniciado em http://localhost:8000")
    server.serve_forever()
