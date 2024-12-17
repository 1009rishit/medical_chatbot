# medical_chatbot
# TO CREATE AN ENVIRONMENT:
1. conda create -n mchatbot python=3.8 -y
2. conda activate mchatbot
# INSTALL ALL THE REQUIRE MODULE 
pip install -r requirements.txt
# LOAD THE DOCUMENTS 

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents

# CREATE TEXT CHUNKS
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
# INITIALIZING EMBEDDING MODEL

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# STORE THE EMBEDDING IN THE CHROMA VECTOR DATABASE

vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings,persist_directory="./chroma_db")

# SET LLM 
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# ADD CONVERSATIONAL BUFFER MEMORY
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)