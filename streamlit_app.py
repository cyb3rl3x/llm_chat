import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Título do aplicativo
st.title("Consulta à LGPD")

# Upload de arquivos PDF
pdf_folder_path = "biblioteca/"
uploaded_files = st.file_uploader("Carregue os arquivos PDF aqui", accept_multiple_files=True, type=['pdf'])

# Instância do modelo Ollama
model_name = 'cnmoro/mistral_7b_portuguese:q2_K'
llm = ChatOllama(model=model_name)

print(f"ARQUIVOS CARREGADOS: {uploaded_files}")

# Carregamento e processamento dos documentos
all_documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        print(f'ARQUIVO {uploaded_file}')
        loader = UnstructuredPDFLoader(uploaded_file)
        document = loader.load()
        all_documents.append(document)
        st.write(f"Documento carregado: {uploaded_file.name}")

# Entrada de pergunta
question = st.text_input("Digite sua pergunta sobre a LGPD:")
if question:
    # Geração da resposta usando o modelo Ollama
    response = llm.generate_response(question)
    st.write("Resposta:", response)
