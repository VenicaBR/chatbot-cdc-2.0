from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Carrega variáveis do .env (apenas em desenvolvimento)
load_dotenv()

# CONFIGURAÇÃO PARA PRODUÇÃO E DESENVOLVIMENTO
if os.getenv('FLASK_ENV') == 'production':
    # EM PRODUÇÃO - caminhos relativos
    BASE_DIR = Path(__file__).parent
    frontend_dir = BASE_DIR / "static"
    faiss_dir = BASE_DIR / "faiss_index"
    documentos_dir = BASE_DIR / "documentos"
    
    print("🌐 MODO PRODUÇÃO")
    print(f"📁 Base: {BASE_DIR}")
    print(f"🎨 Frontend: {frontend_dir}")
    print(f"🗂️ FAISS: {faiss_dir}")
    
else:
    # EM DESENVOLVIMENTO - caminhos relativos à pasta atual
    BASE_DIR = Path(__file__).parent
    frontend_dir = BASE_DIR / "static"
    faiss_dir = Path("C:/chatbot_cdc/faiss_index")
    documentos_dir = BASE_DIR / "documentos"
    
    print("🔧 MODO DESENVOLVIMENTO")
    print(f"📁 Base: {BASE_DIR}")
    print(f"🎨 Frontend: {frontend_dir}")
    print(f"🗂️ FAISS: {faiss_dir}")

print("✅ CONFIGURAÇÕES FINAIS:")
print(f"🎨 Frontend: {frontend_dir} ({'✅' if frontend_dir.exists() else '❌'})")
print(f"📄 index.html: {'✅' if (frontend_dir / 'index.html').exists() else '❌'}")
print(f"🗂️ FAISS: {faiss_dir} ({'✅' if faiss_dir.exists() else '❌'})")
print(f"📚 Documentos: {documentos_dir} ({'✅' if documentos_dir.exists() else '❌'})")

app = Flask(__name__, static_folder=str(frontend_dir), static_url_path='')
CORS(app)

# Verifica API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY não encontrada!")
    print("Configure a variável de ambiente")
    if os.getenv('FLASK_ENV') != 'production':
        exit(1)

client = openai.OpenAI(api_key=api_key) if api_key else None
embedding = OpenAIEmbeddings(openai_api_key=api_key) if api_key else None

# Carrega ou cria índice FAISS
db = None
try:
    if faiss_dir.exists() and list(faiss_dir.glob("*.faiss")):
        print("📖 Carregando índice FAISS existente...")
        db = FAISS.load_local(
            folder_path=str(faiss_dir),
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        print("✅ Índice FAISS carregado!")
    
    elif os.getenv('FLASK_ENV') == 'production':
        print("🔄 Criando índice FAISS em produção...")
        
        # Em produção, cria o índice na inicialização
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        if documentos_dir.exists():
            loader = DirectoryLoader(
                str(documentos_dir), 
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            docs = loader.load()
            
            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = text_splitter.split_documents(docs)
                
                db = FAISS.from_documents(chunks, embedding)
                
                # Salva o índice
                faiss_dir.mkdir(parents=True, exist_ok=True)
                db.save_local(str(faiss_dir))
                print(f"✅ Índice criado com {len(chunks)} chunks")
            else:
                print("❌ Nenhum documento encontrado")
        else:
            print("❌ Pasta de documentos não encontrada")
    
    else:
        print("⚠️ Índice FAISS não encontrado - execute build_index.py primeiro")
        
except Exception as e:
    print(f"❌ Erro ao carregar/criar FAISS: {e}")

@app.route('/')
def home():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return jsonify({
            "erro": "Frontend não encontrado",
            "frontend_path": str(frontend_dir),
            "arquivos_disponiveis": [f.name for f in frontend_dir.glob("*")] if frontend_dir.exists() else []
        }), 404

@app.route('/<path:path>')
def static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError:
        return jsonify({"erro": f"Arquivo {path} não encontrado"}), 404

@app.route('/perguntar', methods=['POST'])
def perguntar():
    if not client or not db:
        return jsonify({
            "resposta": "❌ Sistema não inicializado. Verifique configurações.",
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
        print(f"📝 Pergunta: {pergunta}")

        # Busca documentos similares
        docs_encontrados = db.similarity_search(pergunta, k=4)
        contexto = "\n\n".join([doc.page_content for doc in docs_encontrados])

        if not contexto.strip():
            return jsonify({
                "resposta": "Não encontrei informações relevantes nos documentos para responder sua pergunta.",
                "sugestoes": [
                    "Tente reformular a pergunta",
                    "Use termos mais específicos do Direito do Consumidor",
                    "Pergunte sobre CDC, garantia, vício do produto, etc."
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

        print(f"✅ Resposta gerada")
        return jsonify({
            "resposta": resposta,
            "sugestoes": []
        })

    except Exception as e:
        print(f"❌ Erro: {e}")
        return jsonify({
            "resposta": "Erro interno. Tente novamente.",
            "sugestoes": []
        })

@app.route('/status', methods=['GET'])
def status():
    arquivos_faiss = []
    if faiss_dir.exists():
        arquivos_faiss = [str(arquivo.name) for arquivo in faiss_dir.glob("*")]
    
    return jsonify({
        "status": "online",
        "ambiente": os.getenv('FLASK_ENV', 'development'),
        "faiss_carregado": db is not None,
        "openai_configurado": bool(api_key),
        "frontend_path": str(frontend_dir),
        "faiss_path": str(faiss_dir),
        "arquivos_faiss": arquivos_faiss,
        "frontend_existe": frontend_dir.exists(),
        "index_html_existe": (frontend_dir / 'index.html').exists()
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para monitoramento de saúde"""
    return jsonify({
        "status": "healthy",
        "timestamp": str(Path(__file__).stat().st_mtime)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.getenv('FLASK_ENV') != 'production'
    
    print("\n" + "="*60)
    print("🚀 INICIANDO CHATBOT CDC")
    print(f"🌐 Ambiente: {'PRODUÇÃO' if not debug else 'DESENVOLVIMENTO'}")
    print(f"🎨 Frontend: {frontend_dir}")
    print(f"🗂️ FAISS: {faiss_dir}")
    print(f"🤖 IA: {'✅' if db else '❌'}")
    print(f"🔑 OpenAI: {'✅' if api_key else '❌'}")
    print(f"🌐 Porta: {port}")
    print(f"🔍 Debug: {debug}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)