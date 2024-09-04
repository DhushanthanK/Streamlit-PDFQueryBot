import os
import io
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import fitz  # PyMuPDF 
from langchain.schema import Document

# Suppress specific FutureWarning about clean_up_tokenization_spaces
import warnings
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
    model="llama-3.1-70b-versatile",  # Consider using a smaller model if available
    temperature=0.5,  # Reduced temperature for simpler responses
    max_tokens=256,  # Reduced max_tokens to limit response length
    timeout=10,
    max_retries=1,
    groq_api_key=API_KEY
)

def load_pdf(pdf_file):
    """Load PDF file, extract text, and create a vector store."""
    pdf_file_io = io.BytesIO(pdf_file.read())
    pdf_document = fitz.open(stream=pdf_file_io, filetype="pdf")
    
    documents = [Document(page_content=pdf_document.load_page(page_num).get_text()) for page_num in range(pdf_document.page_count)]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    database_directory = 'db'
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=database_directory)

    return vectordb  

# Streamlit app setup
st.title("Chat with the PDF!")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    vectordb = load_pdf(uploaded_file)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectordb.as_retriever(
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
            prompt = f"Please answer in detail and most relevant: {user_input}"
            response = chain.invoke({"query": prompt})
            
            if 'result' in response:
                st.chat_message("assistant").markdown(response['result'])
                st.session_state.messages.append({'role': 'assistant', 'content': response['result']})
            else:
                st.chat_message("assistant").markdown("Sorry, I couldn't retrieve an answer.")
        except Exception as e:
            st.chat_message("assistant").markdown("An error occurred: Please try again.")
else:
    st.info("Please upload a PDF file to get started.")