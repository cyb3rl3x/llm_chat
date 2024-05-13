#!/usr/bin/env python
# coding: utf-8

# Importação do Streamlit
import time
import streamlit as st

# Importações necessárias para o processamento de documentos
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
import os

# Importações relacionadas ao processamento de texto e embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Importações relacionadas à recuperação e modelos de chat
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Definição do diretório dos PDFs
pdf_folder_path = "biblioteca/"

st.write(os.listdir(pdf_folder_path))
print(os.listdir(pdf_folder_path))

# Carregamento e processamento de documentos
def load_and_process_documents():
    st.write("Carregando dados...")
    st.write(os.listdir(pdf_folder_path))
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    all_documents = []
    for loader in loaders:
        st.write("Carregando documento bruto..." + loader.file_path)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)
        all_documents.extend(documents)
    return all_documents

# Configuração e uso do modelo de embedding e vector database
def setup_vector_db(all_documents):
    model_name = 'cnmoro/mistral_7b_portuguese:q2_K'
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=OllamaEmbeddings(model=model_name, show_progress=True),
        collection_name='local-rag'
    )
    return vector_db

# Definição da interface do usuário
if st.button("Processar documentos"):
    all_documents = load_and_process_documents()
    vector_db = setup_vector_db(all_documents)
    st.write("Processamento completo e vector database configurado.")

# Interface adicional pode ser adicionada aqui para interações subsequentes com o usuário

# Adicionando interface para perguntas do usuário
    question = st.text_input("Digite sua pergunta:")
    if st.button("Enviar pergunta"):
        # Configuração do modelo de chat
        local_model = 'cnmoro/mistral_7b_portuguese:q2_K'
        llm = ChatOllama(model=local_model)
        
        # Template e recuperação de contexto
        prompt_template = PromptTemplate(
            input_variables=['question'],
            template="""Você é um modelo de linguagem de IA. Sua tarefa é gerar respostas para a pergunta dada usando os documentos recuperados."""
        )
        retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), 
                                                 llm, 
                                                 prompt=prompt_template)
        
        template = """Responda a questão baseado somente no seguinte contexto:{context}"""
        
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context":retriever, "question":RunnablePassthrough()}
            |prompt
            |llm
            |StrOutputParser()
        )

        # Execução do modelo de chat
        response = chain.invoke(question)
        print(f'RESPOSTA: {response}')
        st.write("Resposta:")
        st.write(response)
        time.sleep(100)