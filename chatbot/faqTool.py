import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "FAQ_codcoz_v2.pdf"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
def get_faq_context(question):
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter =  RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(question, k=6)
    
    return results