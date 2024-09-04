import os
import io
import fitz
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import warnings
import shutil

# Suppress specific FutureWarning about clean_up_tokenization_spaces
warnings.simplefilter(action='ignore', category=FutureWarning)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Initialize Groq client with API key from environment variables
API_KEY = os.getenv("groq_API_KEY")

# Set up the LLM with optimized parameters
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5,
    max_tokens=256,
    timeout=10,
    max_retries=1,
    groq_api_key=API_KEY
)

CHROMA_PATH = "chroma"

@st.cache_resource
def initialize_chroma():
    """Initialize and return a Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Clear the existing vector store
    os.makedirs(CHROMA_PATH)  # Recreate the directory

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    return db

def load_documents_from_pdf(pdf_file):
    """Load and split documents from a PDF file."""
    pdf_file_io = io.BytesIO(pdf_file.read())
    pdf_document = fitz.open(stream=pdf_file_io, filetype="pdf")

    documents = [
        Document(page_content=pdf_document.load_page(page_num).get_text())
        for page_num in range(pdf_document.page_count)
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)

    return texts

def add_to_chroma(_chunks):
    """Add chunks to the Chroma database."""
    db = initialize_chroma()
    chunks_with_ids = calculate_chunk_ids(_chunks)
    existing_metadatas = db.get(include=["metadatas"])["metadatas"]
    
    # Extract existing IDs
    existing_ids = set(meta["id"] for meta in existing_metadatas)

    # Filter out new chunks that are not already in the database
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])

def calculate_chunk_ids(chunks):
    """Calculate and assign unique IDs to document chunks."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id != last_page_id:
            current_chunk_index = 0
        else:
            current_chunk_index += 1

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks

def query_rag(query_text):
    """Query the Chroma database and return results based on the input query."""
    db = initialize_chroma()
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt = f"Answer the question based only on the following context:\n\n{context_text}\n\n---\n\nAnswer the question based on the above context: {query_text}"
    response = llm.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response.content, sources

# Streamlit app setup
st.title("Chat with the PDF!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    documents = load_documents_from_pdf(uploaded_file)
    add_to_chroma(documents)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=initialize_chroma().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        ),
        return_source_documents=True
    )

    if 'conversation_chain' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=10)
        st.session_state.conversation_chain = ConversationChain(
            llm=llm,
            memory=st.session_state.memory
        )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_input = st.chat_input("Ask me anything...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        try:
            response_text, sources = query_rag(user_input)
            st.chat_message("assistant").markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
        except Exception as e:
            st.chat_message("assistant").markdown("An error occurred: Please try again.")
else:
    st.info("Please upload a PDF file to get started.")                   
    