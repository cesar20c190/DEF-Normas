import sys
import io
import pdfplumber

# Configura a saída para UTF-8 (importante para acentos em português)
# Isso garante que o 'print' não quebre em terminais Windows
sys.stdout.reconfigure(encoding='utf-8')

def extrair_texto():
    try:
        # 1. Lê os bytes do PDF direto da memória (enviados pelo Node.js via Pipe)
        # sys.stdin.buffer lê bytes brutos, não string
        input_stream = sys.stdin.buffer.read()
        
        if not input_stream:
            # Se não chegou nada, imprime erro (mas não quebra o script com exceção)
            print("ERRO: Nenhum dado recebido no script Python.", file=sys.stderr)
            return

        texto_completo = []

        # 2. Usa o pdfplumber para abrir o binário da memória
        with pdfplumber.open(io.BytesIO(input_stream)) as pdf:
            if len(pdf.pages) == 0:
                print("AVISO: O PDF não tem páginas.", file=sys.stderr)
                return

            for i, page in enumerate(pdf.pages):
                # Extrai texto da página
                texto_pagina = page.extract_text()
                
                if texto_pagina:
                    texto_completo.append(texto_pagina)
                else:
                    # Se a página não tiver texto (ex: imagem escaneada), avisa no log de erro
                    print(f"AVISO: Página {i+1} não contém texto selecionável (pode ser imagem).", file=sys.stderr)
        
        # 3. Imprime o texto final para o Node.js capturar
        # Se a lista estiver vazia, significa que o PDF é só imagem
        if not texto_completo:
            print("[O arquivo PDF parece ser uma imagem digitalizada ou está vazio. Nenhum texto foi extraído.]")
        else:
            print("\n".join(texto_completo))

    except Exception as e:
        # Se der qualquer erro crítico, imprime no canal de erro (stderr)
        # O Node.js vai ler isso e saber que falhou
        print(f"Erro Crítico no Python: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    extrair_texto()