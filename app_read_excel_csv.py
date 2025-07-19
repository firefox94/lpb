import fitz  # PyMuPDF
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# 1. load your document
# excel_path = "./movie_reviews.csv" # your document path
# df = pd.read_csv(excel_path)

st.header("Upload your document")
uploaded_file = st.file_uploader("Choose an Excel file or a CSV file", type=["xlsx", "csv"])

documents = []
ids = []

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file,engine='openpyxl') if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    for col in df.columns:
        df[col] = df[col].astype(str)  # Ensure all columns are strings for consistency
    st.write('### Data Preview')
    st.dataframe(df.head())

    for i, row in df.iterrows():
        metadata = {col: row[col] for col in df.columns}
        doc = Document(
            page_content=" ".join(str(row[col]) for col in df.columns),
            metadata=metadata,
            id=str(i)
        )
        ids.append(str(i))
        documents.append(doc)
    #2. create embedding instance
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    #3. create chroma db to store your document
    db_location = "./chroma_db_excel" # location where you wanna store your db
    vector_store = Chroma(
        collection_name="your-document",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    #4. store document in vector store and keep renewing your db whenever you run this app
    vector_store.add_documents(documents=documents, ids=ids)

    #5. create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # set k base on how many docs to return

    #6. setup streamlit interface
    st.set_page_config(page_title="Streaming Agent", page_icon="🤖")
    st.title("VietHa-superSQL 🕵️‍♀️")

    #7. define a function to get the response from your agent
    def get_response(query, chat_history):
        template = """
        You are an expert in so many domains, especialty good at generate sql scripts for query data.
        Chat history: {chat_history}
        Here are the database: {context}
        Here is the question to generate sql query: {query}
        """

        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model="qwen3:8b")
        context = retriever.invoke(query)
        chain = prompt | model
        
        return chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "query": query
        })
        
    #8. build conversation flow
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.markdown(message.content)
        else:
            st.warning(f"Unsupported message type {type(message)}")
            
    #9. user input query
    query = st.chat_input("Which data do you want to query?")
    if query is not None and query != "":
        st.session_state.chat_history.append(HumanMessage(query))
        
        with st.chat_message("human"):
            st.markdown(query)
        
        with st.chat_message("ai"):
            ai_response = get_response(query,st.session_state.chat_history)
            st.markdown(ai_response)
            
        st.session_state.chat_history.append(AIMessage(content=ai_response))
else:
    st.warning("Please upload a file to proceed.")
