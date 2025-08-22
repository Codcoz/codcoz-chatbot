from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate  # <-- IMPORT NECESSÁRIO

# Carrega as variáveis de ambiente
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Inicializa o modelo LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Pasta com os arquivos
pasta_arquivos = os.path.join(os.path.dirname(__file__), "arquivos")

# Lista os arquivos da pasta
try:
    if os.path.isdir(pasta_arquivos):
        arquivos_disponiveis = [
            os.path.join(pasta_arquivos, nome)
            for nome in os.listdir(pasta_arquivos)
            if os.path.isfile(os.path.join(pasta_arquivos, nome))
        ]
    else:
        print(f"Pasta '{pasta_arquivos}' não encontrada.")
        arquivos_disponiveis = []
except Exception as e:
    print(f"Erro ao listar arquivos: {e}")
    arquivos_disponiveis = []

# Nomes dos arquivos
nomes_exibicao = [os.path.basename(arquivo) for arquivo in arquivos_disponiveis]
nomes_validos = set(nomes_exibicao)

# Função segura para leitura
def ler_arquivo(nome_arquivo: str) -> str:
    if nome_arquivo not in nomes_validos:
        return f"Erro: O arquivo '{nome_arquivo}' não está na lista de arquivos disponíveis."
    caminho_completo = os.path.join(pasta_arquivos, nome_arquivo)
    try:
        with open(caminho_completo, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Erro ao ler o arquivo '{nome_arquivo}': {e}"

# Ferramenta
tools = [
    Tool(
        name="ler_arquivo",
        func=ler_arquivo,
        description=f"Lê o conteúdo de um arquivo de texto listado em: {', '.join(nomes_exibicao)}"
    )
]

# Memória e prompt
memory = ConversationBufferMemory(memory_key="chat_history")
mensagem_inicial = f"""Você é um especialista em gêneros musicais.
Use a ferramenta 'ler_arquivo' para obter informações sobre estilos musicais, origens e artistas.
NUNCA invente nomes de arquivos. Use somente os arquivos existentes:

Arquivos disponíveis: {', '.join(nomes_exibicao)}"""
memory.chat_memory.add_user_message(mensagem_inicial)

# Inicializa agente
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)


# rotina para enviar pergunta ao modelo
def generate_bot_reply(pergunta_usuario: str) -> str:
    try:
        resposta_agente = agent.run(pergunta_usuario)
        return resposta_agente.strip()
    except Exception as e:
        print(f"Ocorreu um erro ao executar o agente: {e}")
    try:
        llm_juiz = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=api_key)

        prompt_juiz_template = """Você é um juiz de IA. Avalie se a seguinte afirmação é correta
    (SIM ou NAO) e justifique: "{afirmacao}"."""
        prompt_juiz = ChatPromptTemplate.from_template(prompt_juiz_template)

        # Formata o prompt com a resposta do agente
        prompt_formatado = prompt_juiz.format_messages(afirmacao=resposta_agente)

        # Chama o modelo com o prompt já formatado
        output_juiz = llm_juiz(prompt_formatado)
        avaliacao_juiz = output_juiz.content

        print(f"\nAvaliação do Juiz (Gemini Pro):\n{avaliacao_juiz}")
        if "NAO" in avaliacao_juiz.upper():
            print("\n O Juiz identificou uma possível alucinação!")
        else:
            print("\n O Juiz considerou a informação factual.")
    except Exception as e:
        print(f"Ocorreu um erro ao executar o juiz: {e}")
        return "Ocorreu um erro interno ao processar sua pergunta."



