import express from 'express'
import cors from 'cors'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import 'dotenv/config'

function requireEnv(name) {
  const value = process.env[name]
  if (!value || value.trim().length === 0) {
    throw new Error(`Variavel de ambiente ausente: ${name}`)
  }
  return value
}

const OPENAI_API_KEY = requireEnv('OPENAI_API_KEY')
const SUPABASE_URL = requireEnv('SUPABASE_URL')
const SUPABASE_SERVICE_ROLE_KEY = requireEnv('SUPABASE_SERVICE_ROLE_KEY')

const PORT = Number.parseInt(process.env.PORT ?? '3001', 10)
const CLIENTE_ATUAL = process.env.CLIENTE_ATUAL ?? 'dpe-ba'
const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL ?? 'gpt-4o-mini'
const OPENAI_EMBED_MODEL = process.env.OPENAI_EMBED_MODEL ?? 'text-embedding-3-small'
const OPENAI_TIMEOUT_MS = Number.parseInt(process.env.OPENAI_TIMEOUT_MS ?? '45000', 10)
const OPENAI_MAX_RETRIES = Number.parseInt(process.env.OPENAI_MAX_RETRIES ?? '2', 10)
const OPENAI_RETRY_ATTEMPTS = Number.parseInt(process.env.OPENAI_RETRY_ATTEMPTS ?? '3', 10)
const SEARCH_MATCH_THRESHOLD = Number.parseFloat(process.env.SEARCH_MATCH_THRESHOLD ?? '0.4')
const SEARCH_MATCH_COUNT = Number.parseInt(process.env.SEARCH_MATCH_COUNT ?? '10', 10)
const CONTEXT_MAX_DOCS = Number.parseInt(process.env.CONTEXT_MAX_DOCS ?? '12', 10)
const CONTEXT_MAX_CHARS = Number.parseInt(process.env.CONTEXT_MAX_CHARS ?? '45000', 10)

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  timeout: OPENAI_TIMEOUT_MS,
  maxRetries: OPENAI_MAX_RETRIES,
})

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function shouldRetryOpenAI(error) {
  const status = error?.status
  const name = error?.name
  const code = error?.code

  return (
    name === 'APIConnectionTimeoutError' ||
    name === 'APIConnectionError' ||
    status === 429 ||
    (typeof status === 'number' && status >= 500) ||
    code === 'ETIMEDOUT' ||
    code === 'ECONNRESET'
  )
}

async function withRetry(label, fn, attempts = OPENAI_RETRY_ATTEMPTS) {
  let lastError = null

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      const retry = shouldRetryOpenAI(error) && attempt < attempts

      if (!retry) {
        throw error
      }

      const waitMs = 1000 * 2 ** (attempt - 1)
      console.warn(
        `[${label}] tentativa ${attempt}/${attempts} falhou (${error.name ?? 'Erro'}). Nova tentativa em ${waitMs}ms.`
      )
      await sleep(waitMs)
    }
  }

  throw lastError
}

function explainError(error) {
  const message = error?.message ?? ''
  if (error?.name === 'APIConnectionTimeoutError' || /timed out/i.test(message)) {
    return 'Timeout ao conectar na OpenAI.'
  }
  if (/statement timeout/i.test(message)) {
    return 'Timeout na consulta vetorial (match_documents).'
  }
  return message || 'Erro interno.'
}

const app = express()
app.use(express.json({ limit: '10mb' }))
app.use(cors())
app.use(express.static('public'))

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'consulta-ia' })
})

app.post('/api/consulta', async (req, res) => {
  try {
    const pergunta = (req.body?.pergunta ?? '').trim()

    if (!pergunta) {
      return res.status(400).json({ erro: 'Pergunta vazia.' })
    }

    const embeddingResponse = await withRetry('embeddings', () =>
      openai.embeddings.create({
        model: OPENAI_EMBED_MODEL,
        input: pergunta,
      })
    )

    const vetorPergunta = embeddingResponse.data[0].embedding

    const { data: documentos, error } = await supabase.rpc('match_documents', {
      query_embedding: vetorPergunta,
      match_threshold: SEARCH_MATCH_THRESHOLD,
      match_count: SEARCH_MATCH_COUNT,
      filtro_cliente_id: CLIENTE_ATUAL,
      query_text: pergunta,
    })

    if (error) {
      throw new Error(`Erro SQL: ${error.message}`)
    }

    const docs = documentos ?? []
    const contextoBanco = docs.slice(0, CONTEXT_MAX_DOCS).map((d) => d.content).join('\n\n---\n\n').slice(0, CONTEXT_MAX_CHARS)
    const fontes = [...new Set(docs.map((d) => `${d.source?.toUpperCase()}: ${d.title || 'S/ Titulo'} (ID: ${d.external_id})`))]

    if (!contextoBanco) {
      return res.json({
        resposta: 'Nao encontrei informacoes relevantes na base para essa pergunta.',
        fontes: [],
      })
    }

    const chatResponse = await withRetry('chat', () =>
      openai.chat.completions.create({
        model: OPENAI_CHAT_MODEL,
        messages: [
          {
            role: 'system',
            content:
              'Voce e um assistente juridico da DPE-BA. Responda apenas com base no contexto fornecido e cite limites quando nao houver base suficiente.',
          },
          {
            role: 'user',
            content: `CONTEXTO:\n${contextoBanco}\n\nPERGUNTA:\n${pergunta}`,
          },
        ],
        temperature: 0.1,
      })
    )

    return res.json({
      resposta: chatResponse.choices[0]?.message?.content ?? 'Sem resposta.',
      fontes,
    })
  } catch (error) {
    console.error('[consulta] erro:', error)
    return res.status(500).json({ erro: explainError(error) })
  }
})

app.listen(PORT, () => {
  console.log(`Consulta IA disponivel em http://localhost:${PORT}/consulta.html`)
})
