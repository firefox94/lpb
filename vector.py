from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

df = pd.read_csv("movie_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "chroma_db"
documents = []
ids = []

for i,row in df.iterrows():
    doc = Document(
        page_content=row['movie_title']+" "+row['movie_info'],
        metadata={'rating':row['rating'],'date':row['in_theaters_date'],'director':row['directors']},
        id=str(i)
    )
    ids.append(str(i))
    documents.append(doc)
    
vector_store = Chroma(
    collection_name='movie_review',
    persist_directory=db_location,
    embedding_function=embeddings
)
vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})