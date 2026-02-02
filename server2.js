import express from 'express'
import cors from 'cors'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import 'dotenv/config'

const app = express()
app.use(express.json())
app.use(cors())

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)

// Filtro de Segurança do Cliente
const CLIENTE_ATUAL = process.env.CLIENTE_ATUAL || 'dpe-ba';

app.post('/api/chat', async (req, res) => {
  try {
    // Recebe pergunta e histórico do frontend
    const { pergunta, historico } = req.body
    
    if (!pergunta) return res.status(400).json({ erro: 'Pergunta vazia.' })

    console.log(`\n📩 [Server 2] Pergunta: "${pergunta}"`)

    // 1. Gerar Vetor da Pergunta
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: pergunta,
    })
    const vetorPergunta = embeddingResponse.data[0].embedding

    // 2. Buscar no Banco (com filtro de cliente)
    const { data: documentos, error } = await supabase.rpc('match_documents', {
      query_embedding: vetorPergunta,
      match_threshold: 0.40,
      match_count: 10, // Aumentei para 10 para ter mais chance de achar o artigo exato
      filtro_cliente_id: CLIENTE_ATUAL
    })

    if (error) {
        console.error("Erro Supabase:", error);
        throw error;
    }

    // 3. Preparar Contexto (Aqui estava o erro antes, agora corrigido)
    const contexto = documentos && documentos.length > 0 
      ? documentos.map(d => d.content).join('\n\n---\n\n')
      : "Nenhum documento específico encontrado.";

    // Prepara lista de fontes para exibir no final
    const fontesUnicas = documentos ? [...new Set(documentos.map(d => `${d.source.toUpperCase()}: ${d.title || 'S/ Título'} (ID: ${d.external_id})`))] : [];

    // 4. Montagem da Inteligência (Prompt Jurídico Sênior)
    const chatResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { 
          role: "system", 
          content: `
          ATUAÇÃO:
          Você é um Consultor Jurídico Sênior da Defensoria Pública da Bahia (DPE-BA). 
          Sua função é fornecer respostas estritamente técnicas baseadas nos documentos fornecidos.

          DIRETRIZES RÍGIDAS DE RESPOSTA:
          1. CITAÇÃO OBRIGATÓRIA: Toda afirmação deve vir acompanhada da fonte entre parênteses. Exemplo: "O prazo é de 15 dias (Lei Complementar 26, Art. 12)".
          2. ESTRUTURA:
             - Inicie com uma resposta direta à dúvida.
             - Em seguida, crie um tópico "Fundamentação Legal" e liste os artigos relevantes.
             - Se houver prazos, destaque-os em **negrito**.
          3. SEM ALUCINAÇÃO:
             - Se o contexto não tiver a resposta, diga explicitamente: "A informação solicitada não consta nos documentos consultados."
             - Não use seu conhecimento externo para inventar leis que não estão no contexto.
             - Na constução da resposta dê prioridade aos dados dos documentos informados.
          ` 
        },
        // Injeta o histórico da conversa (Memória)
        ...(historico || []), 
        // Injeta o contexto e a pergunta atual
        { 
          role: "user", 
          content: `
          CONTEXTO DOS DOCUMENTOS :
          ${contexto}
          
          PERGUNTA DO PROCURADOR: 
          ${pergunta}
          ` 
        }
      ],
      temperature: 0, // Zero criatividade para garantir fidelidade
    })

    const respostaIA = chatResponse.choices[0].message.content

    res.json({
      resposta: respostaIA,
      fontes: fontesUnicas
    })

  } catch (erro) {
    console.error("Erro no Server 2:", erro)
    res.status(500).json({ erro: "Erro interno no processamento." })
  }
})

const PORT = process.env.PORT || 3001
app.listen(PORT, () => {
  console.log(`\n⚖️  SERVER JURÍDICO (MEMÓRIA + CITAÇÕES) RODANDO NA PORTA ${PORT}!`)
})
