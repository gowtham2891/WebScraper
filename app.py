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

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.environ['GROQ_API_KEY']

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
        # Log error to terminal and provide a user-friendly message
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
        # Log error to terminal and provide a user-friendly message
        log_interaction("SYSTEM", "error", f"Error processing user input. Error: {e}")
        return "⚠️ An error occurred while processing your request. Please try again."

# Function to log user interactions in the terminal
def log_interaction(username, interaction_type, content):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "type": interaction_type,
        "content": content
    }
    # Print the log entry to the terminal, including URLs or other content
    print(f"[{log_entry['timestamp']}] {log_entry['username']} - {log_entry['type']}: {log_entry['content']}")

# App configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# Custom CSS to match the style from the image you uploaded
st.markdown("""
<style>
    /* General styling for chat bubbles */
    .chat-message {
        padding: 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
        max-width: 80%;
    }

    /* User message styling */
    .user-message {
        background-color: #007bff;  /* Blue background for user messages */
        color: white;  /* White text for user messages */
        text-align: right;  /* Align text to the right */
        margin-left: auto;  /* Push user messages to the right */
    }

    /* AI message styling */
    .bot-message {
        background-color: #f1f1f1;  /* Gray background for AI messages */
        color: black;  /* Black text for AI messages */
        text-align: left;  /* Align text to the left */
        margin-right: auto;  /* Push AI messages to the left */
    }

    /* Thinking placeholder styling */
    .thinking {
        color: #999999; 
        font-style: italic;
        text-align: center;
    }

    /* Clear Chat button styling */
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
            # Log the user's question
            st.session_state.chat_log.insert(0, ('user', user_input))
            log_interaction(st.session_state.username, "user_question", user_input)

            # Display thinking animation
            thinking_placeholder = st.empty()
            for i in range(3):
                thinking_placeholder.markdown(f"<div class='thinking'>🤔 AI is thinking{'.' * (i+1)}</div>", unsafe_allow_html=True)
                time.sleep(0.5)

            # Get AI response
            response = process_user_input(st.session_state.vectors, user_input)
            thinking_placeholder.empty()

            # Log and display the AI's response
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
