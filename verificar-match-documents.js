import 'dotenv/config'
import OpenAI from 'openai'
import { createClient } from '@supabase/supabase-js'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 30000,
  maxRetries: 0,
})

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)

const pergunta = process.argv.slice(2).join(' ').trim() || 'qual o conteudo da portaria 68 de 2007?'
const loops = Number.parseInt(process.env.MATCH_DOC_BENCH_LOOPS ?? '5', 10)
const matchCount = Number.parseInt(process.env.SEARCH_MATCH_COUNT ?? '10', 10)
const threshold = Number.parseFloat(process.env.SEARCH_MATCH_THRESHOLD ?? '0.4')
const clienteId = process.env.CLIENTE_ATUAL ?? 'dpe-ba'

async function main() {
  console.log(`Pergunta: ${pergunta}`)
  console.log(`Loops: ${loops} | match_count: ${matchCount} | threshold: ${threshold} | cliente: ${clienteId}`)

  const emb = await openai.embeddings.create({
    model: process.env.OPENAI_EMBED_MODEL ?? 'text-embedding-3-small',
    input: pergunta,
  })
  const vetor = emb.data[0].embedding

  const times = []
  for (let i = 1; i <= loops; i += 1) {
    const t0 = Date.now()
    const { data, error } = await supabase.rpc('match_documents', {
      query_embedding: vetor,
      match_threshold: threshold,
      match_count: matchCount,
      filtro_cliente_id: clienteId,
      query_text: pergunta,
    })
    const dt = Date.now() - t0
    times.push(dt)

    if (error) {
      console.log(`#${i}: ERRO ${dt}ms | ${error.message}`)
    } else {
      console.log(`#${i}: OK   ${dt}ms | rows=${data?.length ?? 0}`)
    }
  }

  const ok = times.length > 0
  if (ok) {
    const avg = Math.round(times.reduce((a, b) => a + b, 0) / times.length)
    const min = Math.min(...times)
    const max = Math.max(...times)
    console.log(`Resumo tempo: avg=${avg}ms | min=${min}ms | max=${max}ms`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
