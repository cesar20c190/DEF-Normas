import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai' 
import 'dotenv/config'

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)

// A NOVA CHAVE DE SEGURANÇA
const CLIENTE_ATUAL = 'dpe-ba';

async function perguntar(perguntaUsuario) {
  console.log(`\n🔍 Pergunta: "${perguntaUsuario}"`)
  
  // 1. Gerar vetor da pergunta
  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: perguntaUsuario,
  })
  const vetorPergunta = embeddingResponse.data[0].embedding

  // 2. Buscar no banco (AGORA COM O FILTRO OBRIGATÓRIO)
  const { data: documentos, error } = await supabase.rpc('match_documents', {
    query_embedding: vetorPergunta,
    match_threshold: 0.40,
    match_count: 20,
    filtro_cliente_id: CLIENTE_ATUAL // <--- Sem isso, o banco bloqueia
  })

  if (error) { console.error("Erro SQL:", error); return }

  if (!documentos || documentos.length === 0) {
    console.log("❌ Nada encontrado na base deste cliente.")
    return
  }

  console.log(`📊 Encontrei ${documentos.length} trechos relevantes.`)

  // 3. Montar resposta
  const contexto = documentos.map(d => d.content).join('\n\n---\n\n')

  const chatResponse = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { 
        role: "system", 
        content: `Você é um Assistente Jurídico da DPE-BA. Responda com base no contexto.` 
      },
      { role: "user", content: `CONTEXTO:\n${contexto}\n\nPERGUNTA: ${perguntaUsuario}` }
    ],
    temperature: 0.1
  })

  console.log("\n📝 RESPOSTA:")
  console.log("---------------------------------------------------")
  console.log(chatResponse.choices[0].message.content)
  console.log("---------------------------------------------------")
  
  // Mostra as fontes usadas (para você conferir)
  const fontes = [...new Set(documentos.map(d => `${d.source}: ${d.title} (ID: ${d.external_id})`))]
  console.log("\n📚 Fontes:", fontes)
}

// Teste rápido
perguntar("Existe alguma norma sobre home office?")