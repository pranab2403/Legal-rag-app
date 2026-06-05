import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import os

st.set_page_config(page_title="⚖️ Legal AI Assistant", layout="wide")

st.title("⚖️ Legal AI Assistant")
st.caption("Upload PDF & ask legal questions")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

persist_directory = "./chroma_db"

# Sidebar
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

def create_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="llama3"),
        persist_directory=persist_directory
    )
    db.persist()
    return db

# Load/Create DB
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded!")
    vectorstore = create_db("temp.pdf")

elif os.path.exists(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="llama3")
    )
else:
    st.warning("Upload a PDF to start")
    vectorstore = None

# Model
llm = OllamaLLM(model="llama3")

if vectorstore:
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever()
    )

    st.subheader("💬 Chat")

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            result = qa.invoke({"query": user_input})
            answer = result["result"]

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []