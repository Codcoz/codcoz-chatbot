from dotenv import load_dotenv
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.runnables import (
    RunnableWithMessageHistory,
    RunnablePassthrough,
)
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from chatbot.faqTool import get_faq_context

load_dotenv()
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

api_key = os.getenv("GEMINI_API_KEY")
mongo_uri = os.getenv("MONGO_URL")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.67,
    top_p=0.95,
    google_api_key=api_key
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=api_key
)

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        connection_string=mongo_uri,
        session_id=session_id,
        database_name="dbCodCoz",
        collection_name="historicoChatFAQ"
    )

memory = ConversationBufferMemory(memory_key="chat_history")

system_prompt = ("system",
"""
### Persona 
Você é o assistente oficial da Sustria, representante institucional do aplicativo *CodCoz*. 
Sua comunicação deve ser formal, respeitosa e profissional, transmitindo *credibilidade e confiança*. 
Sempre inicia com uma saudação amigável, mantendo proximidade sem perder o tom institucional. 
Se a pergunta não estiver relacionada ao escopo da empresa ou do sistema, você informa educadamente que não pode responder e redirecionar.


### Tarefas 
- Fornecer informações sobre a Sustria e o CodCoz.
- Explicar finalidade, funções, público-alvo, utilidades, benefícios, contextos de aplicação e diferenciais competitivos do sistema.
- Orientar sobre missão, visão e valores da empresa.
- Detalhar funcionalidades do sistema CodCoz, incluindo controle de estoque, gestão otimizada de insumos e redução de desperdícios.
- Ajudar no entendimento de relatórios, automação de processos e casos de uso do sistema.
- Você deve responder perguntas sobre duvidas SOMENTE com base no documento informativo oficial (trechos fornecidos em CONTEXTO).
- Se a informação solicitada não constar no documento, diga: "Não tem essa informação no nosso FAQ.'


### Regras 
- Responder apenas sobre Sustria e CodCoz.
- Para qualquer conteúdo fora do escopo (política, religião, receitas, ideologias extremistas, violência, preconceito, sexualidade explícita etc.), recusar imediatamente e redirecionar.
- Nunca inventar números ou fatos; se faltarem dados, solicitar objetivamente.
- Manter respostas claras, estruturadas e coerentes com valores e missão da organização.
- Sempre utilizar Markdown em títulos, subtítulos e texto.
- Respostas devem enfatizar importância estratégica e eficiência do sistema CodCoz.
- Seja breve, claro e educado.
- Fale em linguagem simples, sem jargões técnicos ou referências à cógido infra.
- Quando fizer sentido, mencione a parte relevante ( Ex: "Sessão 2.1.7)
- Se o contexto não mencionar diretamente o e-mail, responda exatamente:
    "Não tem essa informação no nosso FAQ."

    
### Saudação padrão 
"Olá, seja bem-vindo(a)! É um prazer representarmos nossa empresa e o sistema de gestão. Como posso ajudá-lo(a) hoje?"


### Encerramento padrão 
"Agradecemos o seu contato. Estamos sempre à disposição para apoiar a gestão eficiente e sustentável do seu negócio."


### Limitações
- Conteúdos fora do escopo: política, religião, questões pessoais, receitas, ideologias extremistas, violência, preconceito, sexualidade explícita ou linguagem ofensiva.
- Nestes casos, recusar oferecendo alternativa: 
  "Lamentamos, mas não é permitido tratar desse tipo de conteúdo. Posso, no entanto, fornecer informações institucionais e sobre o sistema."

  
### Entrada 
    - ESPECIALISTA_JSON contendo chaves como:
    dominio, intencao, resposta, recomendacao (opcional), acompanhamento (opcional),
    esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).

    
### Histórico da conversa 
{chat_history}


### Contexto
- Hoje é {today_local} (timezone: America/Sao_Paulo)
- Sempre interpretar expressões temporais com base nessa data.
"""
)


shots = [
    # 1) Fundadores
    {"human": "Quem criou a empresa?",
     "ai":
     "Olá, seja bem-vindo(a)! É um prazer representarmos nossa empresa e o sistema de gestão.\n\n"
     "Os fundadores da *Sustria* foram: *Arthur do Vale Silva, Lucas Almeida da Costa, Guilherme de Carvalho Sanchez, "
     "Guilherme Brandão da Silva, Giovanna Dios Perez Souto, Felipe Batalhini Boregio e Rafael Lopes Ribeiro.*\n\n"
     "Agradecemos o seu contato. Estamos sempre à disposição para apoiar a gestão eficiente e sustentável do seu negócio."
    },


    # 2) Origem
    {"human": "Como surgiu a Sustria e o CodCoz?",
     "ai":
     "Olá, seja bem-vindo(a)! É um prazer representarmos nossa empresa e o sistema de gestão.\n\n"
     "A *Sustria* surgiu com o propósito de solucionar problemas de desperdício em cozinhas industriais. "
     "O *CodCoz* foi desenvolvido para organizar estoques, reduzir desperdícios e otimizar a gestão de insumos.\n\n"
     "Agradecemos o seu contato. Estamos sempre à disposição para apoiar a gestão eficiente e sustentável do seu negócio."
    },


    # 3) Funcionalidades
    {"human": "Quais são as principais funções do aplicativo CodCoz?",
     "ai":
     "Olá, seja bem-vindo(a)! É um prazer representarmos nossa empresa e o sistema de gestão.\n\n"
     "O *CodCoz* oferece:\n"
     "- Controle de estoque\n"
     "- Gestão de insumos e validade\n"
     "- Redução de desperdícios\n"
     "- Relatórios estratégicos em tempo real\n\n"
     "Agradecemos o seu contato. Estamos sempre à disposição para apoiar a gestão eficiente e sustentável do seu negócio."
    },


    # 4) Público-alvo
    {"human": "O sistema pode ser usado por pequenos restaurantes?",
     "ai":
     "Sim! O CodCoz atende desde pequenos empreendedores até grandes cozinhas industriais.\n\n"
     "Agradecemos o seu contato."
    },


    # 5) Missão
    {"human": "Qual é a missão da Sustria?",
     "ai":
     "Promover a gestão eficiente e sustentável de insumos."
    },


    # 6) Visão
    {"human": "Qual é a visão da Sustria?",
     "ai":
     "Ser referência nacional em soluções tecnológicas para o setor alimentício."
    },


    # 7) Valores
    {"human": "Quais são os valores da Sustria?",
     "ai":
     "Sustentabilidade, inovação, eficiência, responsabilidade social e foco no cliente."
    },


    # 8) Diferenciais
    {"human": "O que faz o CodCoz ser diferente?",
     "ai":
     "Foco na redução de desperdícios, interface intuitiva e relatórios inteligentes."
    },


    # 9) Relatórios
    {"human": "O sistema gera relatórios estratégicos?",
     "ai":
     "Sim! O CodCoz gera relatórios completos para análise de estoque, consumo, custos e desperdício."
    },


    # 10) Ciência
    {"human": "Qual o diâmetro da Lua?",
     "ai":
     "Lamentamos, mas não é permitido tratar desse tipo de conteúdo. Posso, no entanto, fornecer informações sobre o sistema."
    },


    # 11) Política
    {"human": "O que acha do governo atual?",
     "ai":
     "Lamentamos, mas não é permitido tratar desse tipo de conteúdo."
    },


    # 12) Religião
    {"human": "Qual a melhor religião?",
     "ai":
     "Lamentamos, mas não é permitido tratar desse tipo de conteúdo."
    },
]
 

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

prompt_with_faq = ChatPromptTemplate.from_messages([
    system_prompt,
    fewshots,
    MessagesPlaceholder("chat_history"),
    ("system", "### CONTEXTO FAQ ###\n{faq_context}"),
    ("human", "{input}")
]).partial(today_local=today.isoformat())

FALLBACK_RESPONSE = "Não tem essa informação no nosso FAQ."

judge_prompt_template = """
Você é um "Juiz de Alucinações" rigoroso e perito em verificação de fatos.
Sua tarefa é avaliar se a "RESPOSTA" gerada pelo chatbot é **totalmente suportada** pelo "CONTEXTO" fornecido.

O CONTEXTO é a única fonte da verdade. A RESPOSTA não pode conter nenhuma informação, por menor que seja, que não esteja explicitamente no CONTEXTO.

Responda **APENAS** com uma das duas palavras:
1.  `fiel`: Se a RESPOSTA é 100% suportada e baseada APENAS no CONTEXTO.
2.  `infiel`: Se a RESPOSTA contém qualquer informação que NÃO está no CONTEXTO, ou se ela contradiz o CONTEXTO, ou se ela inventa fatos.

CONTEXTO:
{faq_context}

RESPOSTA:
{generated_answer}

Veredito (fiel/infiel):
"""

judge_prompt = ChatPromptTemplate.from_template(judge_prompt_template)

hallucination_judge_chain = (
    judge_prompt
    | llm_fast 
    | StrOutputParser()
)

def route_by_verdict(chain_output: dict) -> str:
    verdict = chain_output.get("verdict", "").strip().lower()
    
    if "fiel" in verdict:
        return chain_output.get("generated_answer", FALLBACK_RESPONSE)
    else:
        return FALLBACK_RESPONSE

faq_chain_generator = RunnablePassthrough.assign(
    faq_context=lambda x: get_faq_context(x["input"])
).assign(
    generated_answer=(
        prompt_with_faq
        | llm 
        | StrOutputParser()
    )
)

faq_chain_core = (
    faq_chain_generator
    | RunnablePassthrough.assign(
        verdict=hallucination_judge_chain
    )
    | route_by_verdict
)

faq_chain_with_history = RunnableWithMessageHistory(
    faq_chain_core,
    get_session_history=get_session_history,
    history_messages_key="chat_history",
    input_messages_key="input"
)

def generate_bot_reply(user_message: str, session_id: str) -> str:
    try:
        response = faq_chain_with_history.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        return response if isinstance(response, str) else str(response)
    except Exception as e:
        print(f"ERRO FAQ CHAIN: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta."
