import { consultarPergunta, explainError } from './_consulta-service.js'

function setCorsHeaders(req, res) {
  const configuredOrigin = process.env.CORS_ORIGIN
  const requestOrigin = req.headers.origin

  if (!configuredOrigin || configuredOrigin === '*') {
    res.setHeader('Access-Control-Allow-Origin', '*')
  } else if (requestOrigin && requestOrigin === configuredOrigin) {
    res.setHeader('Access-Control-Allow-Origin', requestOrigin)
  } else {
    res.setHeader('Access-Control-Allow-Origin', configuredOrigin)
  }

  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')
  res.setHeader('Access-Control-Max-Age', '86400')
  res.setHeader('Vary', 'Origin')
}

function readBody(req) {
  if (!req.body) return {}
  if (typeof req.body === 'string') {
    try {
      return JSON.parse(req.body)
    } catch (_error) {
      return {}
    }
  }
  return req.body
}

export default async function handler(req, res) {
  setCorsHeaders(req, res)

  if (req.method === 'OPTIONS') {
    return res.status(204).end()
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ erro: 'Metodo nao permitido.' })
  }

  try {
    const body = readBody(req)
    const pergunta = (body?.pergunta ?? '').trim()

    if (!pergunta) {
      return res.status(400).json({ erro: 'Pergunta vazia.' })
    }

    const data = await consultarPergunta(pergunta)
    return res.status(200).json(data)
  } catch (error) {
    console.error('[api/consulta] erro:', error)
    return res.status(500).json({ erro: explainError(error) })
  }
}
