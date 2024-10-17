import streamlit as st
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import time
from datetime import datetime
import logging

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.environ['GROQ_API_KEY']

# Set up logging to ensure logs are visible in the Streamlit Cloud logs section
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Optional: Save logs to a file
        logging.StreamHandler()  # Ensure logs are captured in Streamlit's log system
    ]
)

# Function to log user interactions
def log_interaction(username, interaction_type, content):
    log_message = f"{datetime.now().isoformat()} - {username} - {interaction_type}: {content}"
    logging.info(log_message)

# Function to ensure URLs include https://
def ensure_https(url):
    parsed = urlparse(url)
    if parsed.scheme == '':
        return 'https://' + url
    return url

# Function to process the URL and return vector store
def process_url(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(data)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector = FAISS.from_documents(documents, embeddings)
        return vector
    except Exception as e:
        log_interaction("SYSTEM", "error", f"Error processing URL: {url}. Error: {e}")
        return None

# Function to process user input and return a response
def process_user_input(vector, user_input):
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            </context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        result = response["answer"]
        return result
    except Exception as e:
        log_interaction("SYSTEM", "error", f"Error processing user input. Error: {e}")
        return "⚠️ An error occurred while processing your request. Please try again."

# App configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# Add a header or disclaimer at the top
st.markdown("""
# AI Chat Assistant
### Note: This application is proprietary, and the source code is not intended for public access. 
If you need information or support, please reach out through the appropriate channels.
""")

# Custom CSS to style the chat
st.markdown("""
<style>
    .chat-message {
        padding: 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        text-align: right;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f1f1f1;
        color: black;
        text-align: left;
        margin-right: auto;
    }
    .thinking {
        color: #999999; 
        font-style: italic;
        text-align: center;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# User login
if 'username' not in st.session_state:
    st.title("🤖 Welcome to AI Chat Assistant")
    username = st.text_input("Please enter your name to start:")
    if st.button("Start Chatting"):
        if username:
            st.session_state.username = username
            log_interaction(username, "login", "User logged in")
            st.rerun()
        else:
            st.warning("Please enter a name to continue.")
else:
    # Main application
    st.title(f"🤖 AI Chat Assistant - Welcome, {st.session_state.username}!")
    st.markdown(f"Hello {st.session_state.username}! Enter a URL to analyze, then ask questions about its content.")

    # URL input section
    with st.container():
        st.markdown("### 📚 Step 1: Enter a URL to analyze")
        with st.form(key='url_form'):
            url = st.text_input("Enter Your URL", key="url", placeholder="https://example.com")
            submit_url_button = st.form_submit_button(label='🔍 Process URL')

    if submit_url_button and url:
        formatted_url = ensure_https(url)
        with st.spinner("🔄 Processing URL... This may take a moment."):
            vector_store = process_url(formatted_url)
        
        if vector_store is None:
            st.error("⚠️ There was an issue processing the URL. Please make sure the URL is correct and try again.")
        else:
            st.session_state.vectors = vector_store
            st.success("✅ URL processed successfully. You can now ask your questions!")
            log_interaction(st.session_state.username, "process_url", formatted_url)  # Log the URL

    # Initialize session state for chat log
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []

    # Chat interface
    st.markdown("### 💬 Step 2: Ask your questions")
    user_input = st.text_input("Type your question here", key="user_input", placeholder="What would you like to know about the content?")

    if st.button('📤 Send', key='send_button'):
        if user_input and 'vectors' in st.session_state:
            st.session_state.chat_log.insert(0, ('user', user_input))
            log_interaction(st.session_state.username, "user_question", user_input)

            thinking_placeholder = st.empty()
            for i in range(3):
                thinking_placeholder.markdown(f"<div class='thinking'>🤔 AI is thinking{'.' * (i+1)}</div>", unsafe_allow_html=True)
                time.sleep(0.5)

            response = process_user_input(st.session_state.vectors, user_input)
            thinking_placeholder.empty()

            st.session_state.chat_log.insert(0, ('bot', response))
            log_interaction(st.session_state.username, "ai_response", response)
            st.rerun()  # Rerun to update the chat interface
        elif 'vectors' not in st.session_state:
            st.warning("⚠️ Please process a URL first before asking questions.")
        elif not user_input:
            st.warning("⚠️ Please enter a question.")

    # Display chat log
    st.markdown("### 📜 Chat History")
    for author, message in st.session_state.chat_log:
        if author == 'user':
            st.markdown(f"<div class='chat-message user-message'><strong>👤 You:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><strong>🤖 AI:</strong> {message}</div>", unsafe_allow_html=True)

    # Clear chat button
    if st.button('🗑️ Clear Chat History', key='clear_button'):
        st.session_state.chat_log = []
        log_interaction(st.session_state.username, "clear_chat", "Chat history cleared")
        st.rerun()

    # Logout button
    if st.button('🚪 Logout'):
        log_interaction(st.session_state.username, "logout", "User logged out")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
