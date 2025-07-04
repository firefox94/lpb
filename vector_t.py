from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Đọc file .txt
with open("movie_reviews.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Tách văn bản thành nhiều đoạn nhỏ
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
text_chunks = splitter.split_text(text)

# 3. Tạo Document list
documents = []
for i, chunk in enumerate(text_chunks):
    doc = Document(
        page_content=chunk,
        metadata={"source": f"chunk_{i}"}
    )
    documents.append(doc)

# 4. Tạo embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 5. Khởi tạo Chroma vector store
db_location = "chroma_db_txt"
vector_store = Chroma(
    collection_name="txt_movie_review",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# 6. Đưa vào vector store
vector_store.add_documents(documents)

# 7. Tạo retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
