from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Đọc từng trang từ file PDF
pdf_path = "movie_reviews.pdf"
doc = fitz.open(pdf_path)

documents = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    if text.strip():  # bỏ trang trắng
        document = Document(
            page_content=text,
            metadata={"page": page_num + 1}  # đánh số từ 1
        )
        documents.append(document)

doc.close()

# 5. Tạo Embedding
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 6. Tạo Chroma Vector Store
db_location = "chroma_db_pdf"
vector_store = Chroma(
    collection_name="pdf_movie_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# 7. Thêm documents vào vector store
vector_store.add_documents(documents)

# Tạo memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Tạo chain tích hợp retriever + memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=your_llm,                     # ví dụ: ChatOpenAI(), Ollama(), v.v.
    retriever=retriever,              # retriever từ vector store
    memory=memory
)

# 8. Tạo retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})