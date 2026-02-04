import express from 'express'
import cors from 'cors'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import multer from 'multer'
import { spawn } from 'child_process' 
import path from 'path'
import { fileURLToPath } from 'url'
import 'dotenv/config'

// Configuração para caminhos no ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express()
app.use(express.json({ limit: '50mb' })) 
app.use(express.urlencoded({ limit: '50mb', extended: true }))
app.use(cors())

const upload = multer({ storage: multer.memoryStorage() })
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)
const CLIENTE_ATUAL = 'dpe-ba';

// --- FUNÇÃO AUXILIAR: CHAMAR O PYTHON ---
function extrairTextoComPython(bufferDoArquivo) {
    return new Promise((resolve, reject) => {
        // Caminho absoluto para o script python para evitar erros de pasta
        const scriptPath = path.join(__dirname, 'extrator.py');
        
        // Define comando (windows vs linux/mac)
        // Se no seu terminal você usa 'python3', altere aqui.
        const comandoPython = process.platform === "win32" ? "python" : "python3";
        
        const pythonProcess = spawn(comandoPython, [scriptPath]);

        let textoResultado = '';
        let erroResultado = '';

        // Trata erro ao tentar iniciar o processo (ex: python não instalado)
        pythonProcess.on('error', (err) => {
            reject(new Error(`Falha ao iniciar Python: ${err.message}`));
        });

        // Envia o arquivo para o Python via Stdin (Stream)
        try {
            pythonProcess.stdin.write(bufferDoArquivo);
            pythonProcess.stdin.end();
        } catch (err) {
            reject(new Error(`Erro ao escrever no stdin do Python: ${err.message}`));
        }

        // Ouve a resposta do Python
        pythonProcess.stdout.on('data', (data) => {
            textoResultado += data.toString();
        });

        // Ouve erros do script Python
        pythonProcess.stderr.on('data', (data) => {
            erroResultado += data.toString();
        });

        // Quando o Python terminar
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Script Python falhou (Exit Code ${code}): ${erroResultado}`));
            } else {
                resolve(textoResultado);
            }
        });
    });
}

// --- ROTA 1: UPLOAD (VIA PYTHON) ---
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ erro: 'Nenhum arquivo enviado.' })

    console.log(`\n📂 [Upload] Enviando para Python: ${req.file.originalname}`)
    
    // Chama o script Python
    const textoExtraido = await extrairTextoComPython(req.file.buffer);

    if (!textoExtraido || textoExtraido.trim().length === 0) {
        console.warn("⚠️ Python retornou texto vazio.");
        res.json({ texto: "[O PDF foi lido, mas não contém texto selecionável. Pode ser uma imagem escaneada.]" });
        return;
    }

    console.log(`✅ Sucesso! Python extraiu ${textoExtraido.length} caracteres.`);
    res.json({ texto: textoExtraido })

  } catch (erro) {
    console.error("Erro no upload:", erro.message)
    res.status(500).json({ erro: `Falha ao processar PDF: ${erro.message}` })
  }
})

// --- ROTA 2: CHAT (IA) ---
app.post('/api/chat', async (req, res) => {
  try {
    const { pergunta, historico, contexto_arquivo } = req.body
    
    if (!pergunta) return res.status(400).json({ erro: 'Pergunta vazia.' })

    console.log(`\n📩 [Chat] Pergunta: "${pergunta}"`)
    
    if (contexto_arquivo) {
        console.log(`📎 [Anexo] Usando contexto do arquivo (${contexto_arquivo.length} chars).`)
    }

    // 1. Gerar Vetor
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: pergunta,
    })
    const vetorPergunta = embeddingResponse.data[0].embedding

    // 2. Busca no Supabase
    const { data: documentos, error } = await supabase.rpc('match_documents', {
      query_embedding: vetorPergunta,
      match_threshold: 0.40,
      match_count: 10,       
      filtro_cliente_id: CLIENTE_ATUAL,
      query_text: ""         
    })

    if (error) throw error

    // 3. Contexto
    const contextoBanco = documentos && documentos.length > 0 
        ? documentos.map(d => d.content).join('\n\n---\n\n') 
        : "";

    const fontesUnicas = documentos ? [...new Set(documentos.map(d => `${d.source.toUpperCase()}: ${d.title || 'S/ Título'} (ID: ${d.external_id})`))] : [];

    if (!contextoBanco && !contexto_arquivo) {
        return res.json({ 
            resposta: "Não encontrei informações relevantes no banco de dados e nenhum arquivo foi anexado.",
            fontes: []
        })
    }

    // 4. Inteligência
    const chatResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { 
          role: "system", 
          content: `
          ATUAÇÃO: Consultor Jurídico Sênior da DPE-BA.
          OBJETIVO: Responder cruzando o PDF ANEXADO (se houver) com as NORMAS DO BANCO.
          
          DIRETRIZES:
          1. Use o PDF Anexado como fato concreto.
          2. Use o Banco de Dados como base legal.
          3. Cite sempre a fonte.
          ` 
        },
        ...(historico || []), 
        { 
          role: "user", 
          content: `
          === 1. TEXTO DO ARQUIVO ANEXADO ===
          ${contexto_arquivo || "(Nenhum)"}

          === 2. NORMAS DO BANCO DE DADOS ===
          ${contextoBanco || "(Nenhuma)"}
          
          === 3. PERGUNTA ===
          ${pergunta}
          ` 
        }
      ],
      temperature: 0,
    })

    res.json({
      resposta: chatResponse.choices[0].message.content,
      fontes: fontesUnicas
    })

  } catch (erro) {
    console.error("Erro no Server:", erro)
    res.status(500).json({ erro: "Erro interno." })
  }
})

const PORT = 3001
app.listen(PORT, () => {
  console.log(`\n🚀 SERVIDOR HÍBRIDO (NODE + PYTHON) RODANDO NA PORTA ${PORT}!`)
})