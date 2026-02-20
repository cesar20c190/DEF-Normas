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

export default async function handler(req, res) {
  setCorsHeaders(req, res)

  if (req.method === 'OPTIONS') {
    return res.status(204).end()
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ erro: 'Metodo nao permitido.' })
  }

  return res.status(200).json({ ok: true, service: 'consulta-ia' })
}
