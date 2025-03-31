"""
This code demonstrates how to use AWS Bedrock to create a Streamlit application that allows users to interact with PDF documents.
The application performs the following tasks:
1. Data Ingestion: Loads PDF documents from the 'data' directory and splits them into manageable chunks.
2. Vector Embedding and Storage: Uses the FAISS library to create vector embeddings of the document chunks and stores them locally.
3. LLM Integration: Integrates with various Large Language Models (LLMs) such as Llama2 and Claude to generate responses based on user queries.
4. Streamlit Interface: Provides a user-friendly GUI to Q&A with the PDF documents and update the vector store.

Key Functions:
- data_ingestion: Loads and splits PDF documents.
- get_vector_store: Creates and saves vector embeddings.
- get_claude_llm: Initializes the Claude LLM.
- get_llama2_llm: Initializes the Llama2 LLM.
- get_response_llm: Generates responses using the specified LLM and vector store.
- main: Sets up the Streamlit interface and handles user interactions.
"""

import json
import os
import sys
import boto3
import streamlit as st

# We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")

# To find the model_id for embeddings, search under Amazon bedrock -> model catalog ->
# Embeddings -> "Titan Embeddings G1 - Text" -> request access if you don't have.

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # In our testing, Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    # Create the Anthropic Model
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    # Create the Anthropic Model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            # Use "allow_dangerous_deserialization=True" below only if you trust the pickle files in faiss_index in this project
            # which is generally safe if you are generating it. Hence it is safe to use it here.
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()