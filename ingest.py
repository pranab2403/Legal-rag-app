from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

loader = PyPDFLoader("data/constitution_of_india.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="llama3"),
    persist_directory="./chroma_db"
)

db.persist()

print("✅ DB created successfully")