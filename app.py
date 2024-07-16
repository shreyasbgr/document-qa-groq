import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # vector store db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Vector embedding technique

from dotenv import load_dotenv
load_dotenv()

# Load the Groq and GOOGLE API Key from the .env file
groq_key = os.getenv('GROQ_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

st.title("Gamma model Chatbot powered by Groq")
llm=ChatGroq(groq_api_key=groq_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    """
    Initialize the vector store in session state.

    This function is responsible for loading the documents, splitting them into chunks,
    creating the vector store and storing it in the session state.
    """
    # Check if the vector store is already initialized
    if "vectors" not in st.session_state:
        # Initialize the embedding model
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load the documents from the directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()

        # Split the documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create and store the vector store in session state
        st.session_state["vectors"] = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

"""
This app initializes a vector store using the documents in the 'us_census' directory.
It then allows the user to input a question and retrieve the most relevant documents from the vector store.
"""
import time

def vector_embedding():
    """
    Initialize the vector store in session state.

    This function is responsible for loading the documents, splitting them into chunks,
    creating the vector store and storing it in the session state.
    """
    # Check if the vector store is already initialized
    if "vectors" not in st.session_state:
        # Initialize the embedding model
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load the documents from the directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()

        # Split the documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create and store the vector store in session state
        st.session_state["vectors"] = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )


prompt1 = st.text_input("Enter your question you wish to ask from the documents")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store Created")


if prompt1:
    # Create a chain of documents to be combined for question-answering
    documentchain = create_stuff_documents_chain(llm, prompt)
    
    # Check if the vector store is already initialized
    if "vectors" in st.session_state and st.session_state["vectors"] is not None:
        # Create a retriever from the vector store
        retriever = st.session_state["vectors"].as_retriever()
        # Create a chain of retrieval for question-answering
        retrieval_chain = create_retrieval_chain(retriever, documentchain)

        # Measure the time taken for the retrieval
        start = time.process_time()
        # Invoke the chain with the user's question
        response = retrieval_chain.invoke({"input": prompt1})
        # Write the answer to the user's question
        st.write(response["answer"])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                # Write the content of each document
                st.write(doc.page_content)
                # Separate the documents with a line
                st.write("--------------------------------")
    else:
        # Write an error message if the vector store is not created properly
        st.write("Vector Store is not created properly. Please create the Vector Store first.")

