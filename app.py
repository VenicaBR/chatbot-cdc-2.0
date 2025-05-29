from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Carrega variáveis do .env
load_dotenv()

# CAMINHOS AUTOMÁTICOS - detecta a estrutura do projeto
projeto_dir = Path(__file__).parent.parent  # Vai para a pasta raiz do projeto
frontend_dir = projeto_dir / "Front_chatbot"
faiss_dir = Path("C:/chatbot_cdc/faiss_index")

# Tenta encontrar a pasta frontend com diferentes nomes
possible_frontend_dirs = [
    projeto_dir / "Front_chatbot",
    projeto_dir / "frontend", 
    projeto_dir / "Front",
    projeto_dir / "web"
]

for possible_dir in possible_frontend_dirs:
    if possible_dir.exists() and (possible_dir / "index.html").exists():
        frontend_dir = possible_dir
        break

print("Configurações:")
print(f"  Projeto: {projeto_dir}")
print(f"  Frontend: {frontend_dir}")
print(f"  FAISS: {faiss_dir}")
print(f"  Frontend existe: {frontend_dir.exists()}")
print(f"  index.html existe: {(frontend_dir / 'index.html').exists()}")
print(f"  FAISS existe: {faiss_dir.exists()}")

app = Flask(__name__, static_folder=str(frontend_dir), static_url_path='')
CORS(app)

# Verifica API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY não encontrada!")
    exit(1)

client = openai.OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

# Carrega índice FAISS
db = None
try:
    if faiss_dir.exists():
        print("Carregando índice FAISS...")
        db = FAISS.load_local(
            folder_path=str(faiss_dir),
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        print("✅ Índice FAISS carregado!")
    else:
        print(f"❌ Diretório FAISS não encontrado: {faiss_dir}")
        print("Execute build_index_alternativo.py primeiro")
        
except Exception as e:
    print(f"❌ Erro ao carregar FAISS: {e}")
    print("Execute build_index_alternativo.py primeiro")

@app.route('/')
def home():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return jsonify({
            "erro": "index.html não encontrado",
            "frontend_path": str(frontend_dir)
        }), 404

@app.route('/<path:path>')
def static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError:
        return jsonify({"erro": f"Arquivo {path} não encontrado"}), 404

@app.route('/perguntar', methods=['POST'])
def perguntar():
    if db is None:
        return jsonify({
            "resposta": "❌ Sistema não inicializado. Execute build_index_alternativo.py primeiro.",
            "sugestoes": []
        })
    
    try:
        data = request.get_json()
        if not data or not data.get('pergunta'):
            return jsonify({
                "resposta": "Por favor, faça uma pergunta.",
                "sugestoes": []
            })

        pergunta = data.get('pergunta')
        print(f"Pergunta recebida: {pergunta}")

        # Busca documentos similares
        docs_encontrados = db.similarity_search(pergunta, k=4)
        contexto = "\n\n".join([doc.page_content for doc in docs_encontrados])

        if not contexto.strip():
            return jsonify({
                "resposta": "Não encontrei informações relevantes nos documentos para responder sua pergunta.",
                "sugestoes": [
                    "Tente reformular a pergunta",
                    "Use termos mais específicos do Direito do Consumidor",
                    "Consulte sobre CDC, garantia, vício do produto, etc."
                ]
            })

        # Prepara mensagem para OpenAI
        mensagens = [
            {
                "role": "system",
                "content": (
                    "Você é um advogado especialista em Direito do Consumidor brasileiro. "
                    "Responda APENAS com base nos documentos fornecidos abaixo. "
                    "Se a informação não estiver nos documentos, diga que não está disponível. "
                    "Seja claro, objetivo e cite artigos quando possível.\n\n"
                    f"Documentos:\n{contexto}"
                )
            },
            {"role": "user", "content": pergunta}
        ]

        # Chama OpenAI
        resposta = client.chat.completions.create(
            model="gpt-4",
            messages=mensagens,
            temperature=0.3,
            max_tokens=600
        ).choices[0].message.content.strip()

        return jsonify({
            "resposta": resposta,
            "sugestoes": []
        })

    except Exception as e:
        print(f"Erro ao processar pergunta: {e}")
        return jsonify({
            "resposta": "Erro interno. Tente novamente.",
            "sugestoes": []
        })

@app.route('/status', methods=['GET'])
def status():
    # Converte WindowsPath para string para evitar erro de serialização JSON
    arquivos_faiss = []
    if faiss_dir.exists():
        arquivos_faiss = [str(arquivo.name) for arquivo in faiss_dir.glob("*")]
    
    return jsonify({
        "status": "online",
        "faiss_carregado": db is not None,
        "openai_configurado": bool(os.getenv("OPENAI_API_KEY")),
        "frontend_path": str(frontend_dir),
        "faiss_path": str(faiss_dir),
        "arquivos_faiss": arquivos_faiss
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 INICIANDO CHATBOT CDC")
    print(f"📁 Frontend: {frontend_dir}")
    print(f"🗂️  FAISS: {faiss_dir}")
    print(f"🤖 IA: {'✅' if db else '❌'}")
    print("🌐 Acesse: http://localhost:8000")
    print("📊 Status: http://localhost:8000/status")
    print("="*50)
    
    app.run(host='0.0.0.0', port=8000, debug=True)


