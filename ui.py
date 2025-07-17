import os
import warnings
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Title
st.title("Ask Chatbot!")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# File upload
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# Cache vectorstore from PDF
@st.cache_resource(show_spinner="Indexing the uploaded PDF...")
def get_vectorstore_from_path(file_path):
    loaders = [PyPDFLoader(file_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# User prompt input
prompt = st.chat_input("Ask a question about your PDF")

if prompt and uploaded_pdf:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Save uploaded PDF temporarily
        temp_path = "temp_uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # Load Vectorstore
        vectorstore = get_vectorstore_from_path(temp_path)

        # Setup model
        model = "llama3-8b-8192"
        groq_chat = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name=model
        )

        # Setup retrieval chain
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # Get response
        with st.spinner("Thinking..."):
            result = chain({"query": prompt})
            response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

elif prompt and not uploaded_pdf:
    st.warning("Please upload a PDF file first.")
