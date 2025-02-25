# app.py
import streamlit as st
from pages import home, upload, chatbot, study_tools
from database.vectorstore import VectorStore
from agents.content_agent import ContentAgent
from agents.qa_agent import QAAgent
from agents.specialized_agents import CheatSheetAgent, QuizAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Learning Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the file navigation from sidebar
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialize core components if not already in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore(llm_choice="Groq API")

if 'content_agent' not in st.session_state:
    st.session_state.content_agent = ContentAgent(st.session_state.vector_store)

if 'qa_agent' not in st.session_state:
    st.session_state.qa_agent = QAAgent(
        st.session_state.vector_store,
        llm_choice="Groq API",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        hf_key=os.getenv("HF_API_KEY")
    )

# Initialize specialized agents
if 'cheatsheet_agent' not in st.session_state:
    st.session_state.cheatsheet_agent = CheatSheetAgent(
        st.session_state.vector_store,
        st.session_state.qa_agent
    )

if 'quiz_agent' not in st.session_state:
    st.session_state.quiz_agent = QuizAgent(
        st.session_state.vector_store,
        st.session_state.qa_agent
    )

# Navigation
st.sidebar.title("Navigation")
pages = {
    "Home": home.app,
    "Upload Materials": upload.app,
    "Study Tools": study_tools.app,
    "Chatbot": chatbot.app
}
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run the selected page
pages[selection]()


