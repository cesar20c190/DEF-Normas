import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import 'dotenv/config'

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const CLIENTE_ATUAL = 'dpe-ba'

// CONFIGURAÇÃO TURBO
const BATCH_SIZE = 50; // Processa 50 pedaços por vez
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

function fatiarTexto(texto, tamanho, sobreposicao) {
  if (!texto) return [];
  const chunks = [];
  let inicio = 0;
  while (inicio < texto.length) {
    let fim = inicio + tamanho;
    let pedaco = texto.slice(inicio, fim);
    if (fim < texto.length) {
      const ultimoEspaco = pedaco.lastIndexOf(' ');
      if (ultimoEspaco > -1) {
        pedaco = pedaco.slice(0, ultimoEspaco);
        inicio += (ultimoEspaco + 1) - sobreposicao;
      } else {
        inicio += tamanho - sobreposicao;
      }
    } else {
      inicio = fim;
    }
    chunks.push(pedaco);
  }
  return chunks;
}

// Função auxiliar para processar um lote acumulado
async function processarLote(batchItens) {
  if (batchItens.length === 0) return;

  try {
    // 1. Extrai apenas os textos para enviar para a OpenAI
    const textosParaOpenAI = batchItens.map(item => item.content);

    // 2. Chamada em LOTE para a OpenAI (Muito rápido!)
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: textosParaOpenAI,
    });

    // 3. Mescla os vetores recebidos com os dados originais
    const dadosParaSalvar = batchItens.map((item, index) => ({
      cliente_id: CLIENTE_ATUAL,
      source: item.source,
      external_id: item.external_id,
      content: item.content,
      embedding: response.data[index].embedding // O vetor correspondente
    }));

    // 4. Salva tudo no Supabase de uma vez
    const { error } = await supabase.from('doc_embeddings').insert(dadosParaSalvar);

    if (error) throw error;
    
    // Feedback visual simples
    process.stdout.write("."); // Imprime um pontinho a cada lote salvo

  } catch (erro) {
    console.error(`\n❌ Erro ao processar lote:`, erro.message);
  }
}

async function gerarEmbeddingsTurbo() {
  console.log("🚀 Iniciando MODO TURBO (Batching)...");
  
  // -- ETAPA 1: Baixar Documentos --
  let allDocuments = [];
  let from = 0;
  let pageSize = 1000;
  let fetchMore = true;

  console.log("📥 Baixando documentos do Supabase...");
  while (fetchMore) {
    const { data, error } = await supabase
      .from('unified_documents')
      .select('source, external_id, title, ementa, text')
      .range(from, from + pageSize - 1);

    if (error) { console.error(error); break; }
    if (data.length > 0) {
      allDocuments = allDocuments.concat(data);
      from += pageSize;
      console.log(`   ...Baixou +${data.length} docs`);
      if (data.length < pageSize) fetchMore = false;
    } else { fetchMore = false; }
  }

  console.log(`📚 Total de docs: ${allDocuments.length}. Preparando fatias...`);

  // -- ETAPA 2: Processamento em Lotes --
  let buffer = []; // Fila de espera
  let totalChunks = 0;

  console.log("⚡ Processando (cada ponto '.' são 50 vetores salvos):");

  for (const doc of allDocuments) {
    const textoBase = `Título: ${doc.title || ''}\nEmenta: ${doc.ementa || ''}\nTexto: ${doc.text || ''}`.trim();
    if (!textoBase) continue;

    const fatias = fatiarTexto(doc.text || '', CHUNK_SIZE, CHUNK_OVERLAP);
    if (fatias.length === 0) fatias.push(textoBase.substring(0, 1000));

    // Para cada fatia, adiciona na fila (buffer)
    for (const [i, fatia] of fatias.entries()) {
      const conteudoDoChunk = `CONTEXTO:\nID: ${doc.external_id}\nTítulo: ${doc.title}\nEmenta: ${doc.ementa}\n---\nTRECHO ${i + 1}/${fatias.length}:\n${fatia}`.trim();

      buffer.push({
        source: doc.source,
        external_id: doc.external_id,
        content: conteudoDoChunk
      });

      // SE A FILA ENCHEU, MANDA PROCESSAR
      if (buffer.length >= BATCH_SIZE) {
        await processarLote(buffer);
        totalChunks += buffer.length;
        buffer = []; // Limpa a fila
      }
    }
  }

  // Processa o que sobrou na fila (o resto final)
  if (buffer.length > 0) {
    await processarLote(buffer);
    totalChunks += buffer.length;
  }

  console.log(`\n\n🏁 SUCESSO!`);
  console.log(`Total de vetores gerados e salvos: ${totalChunks}`);
}

gerarEmbeddingsTurbo();
