# import streamlit as st
# from agents.qa_agent import QAAgent
# from database.vectorstore import VectorStore
# from agents.content_agent import ContentAgent
# from agents.research_crew import ResearchCrew
# from voice_interface import VoiceInterface
# from memory_store import MemoryStore
# from dotenv import load_dotenv
# import os
# import logging
# import uuid

# logging.getLogger('root').setLevel(logging.ERROR)

# def app():
#     st.title("🤖 Chat with Your Learning Assistant")
#     load_dotenv()

#     # Initialize session ID if not exists
#     if 'session_id' not in st.session_state:
#         st.session_state.session_id = str(uuid.uuid4())

#     # Initialize memory store if not exists
#     if 'memory_store' not in st.session_state:
#         st.session_state.memory_store = MemoryStore()

#     # Initialize voice interface
#     if 'voice_interface' not in st.session_state:
#         st.session_state.voice_interface = VoiceInterface()

#     # Initialize voice input tracker
#     if 'last_input_was_voice' not in st.session_state:
#         st.session_state.last_input_was_voice = False

#     # Initialize components
#     llm_choice = st.sidebar.selectbox(
#         "Choose an LLM:", ["Groq API", "HuggingFace API"]
#     )

#     # Add memory toggle in sidebar
#     use_memory = st.sidebar.checkbox("Use Long-term Memory", value=True)

#     vector_store = VectorStore(llm_choice=llm_choice)
#     content_agent = ContentAgent(vector_store)
#     qa_agent = QAAgent(
#         vector_store,
#         llm_choice=llm_choice,
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#         hf_key=os.getenv("HF_API_KEY"),
#     )

#     research_crew = ResearchCrew(
#         content_agent=content_agent,
#         qa_agent=qa_agent,
#         vector_store=vector_store
#     )

#     # Initialize message history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Create containers
#     chat_container = st.container()
#     input_container = st.container()

#     # Display chat history
#     with chat_container:
#         for message in st.session_state.messages:
#             if message["role"] == "user":
#                 st.chat_message("user").markdown(message["content"])
#             else:
#                 st.chat_message("assistant").markdown(message["content"])

#     # Add input elements
#     with input_container:
#         col1, col2 = st.columns([1, 5])
        
#         # Text input
#         with col2:
#             text_prompt = st.chat_input("Type or press 🎤 to speak...")
#             if text_prompt:
#                 prompt = text_prompt
#                 st.session_state.last_input_was_voice = False

#         # Voice input
#         with col1:
#             if st.button("🎤 Record"):
#                 with st.spinner("Listening..."):
#                     transcribed_text = st.session_state.voice_interface.record_audio()
#                     if transcribed_text:
#                         prompt = transcribed_text
#                         st.session_state.last_input_was_voice = True
#                     else:
#                         st.error("No speech detected. Please try again.")
#                         st.session_state.last_input_was_voice = False
#                         prompt = None

#         # Process input
#         if 'prompt' in locals() and prompt:
#             with chat_container:
#                 # Store user message
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 st.chat_message("user").markdown(prompt)

#                 if use_memory:
#                     # Store in long-term memory
#                     st.session_state.memory_store.store_message(
#                         session_id=st.session_state.session_id,
#                         role="user",
#                         content=prompt
#                     )
#                     # Get relevant history
#                     relevant_history = st.session_state.memory_store.get_relevant_history(
#                         session_id=st.session_state.session_id,
#                         query=prompt
#                     )
#                     # Create context from relevant history
#                     history_context = "\n".join([f"{role}: {content}" for role, content in relevant_history])
#                 else:
#                     history_context = None

#                 try:
#                     # Get response with context
#                     qa_response = qa_agent.get_answer(prompt, history_context)
                    
#                     if qa_response["sources"] and qa_response["sources"] != ["No specific sources"]:
#                         answer = qa_response["answer"]
#                         sources = qa_response["sources"]
#                     else:
#                         with st.spinner("Researching your question..."):
#                             research_response = research_crew.process_query(prompt)
#                             answer = research_response["answer"]
#                             sources = research_response["sources"]

#                     # Format response
#                     sources_display = "\n".join(sources) if sources else "No sources available."
#                     full_response = f"{answer}\n\n**Sources:**\n{sources_display}"
                    
#                     st.chat_message("assistant").markdown(full_response)
#                     st.session_state.messages.append({"role": "assistant", "content": full_response})

#                     if use_memory:
#                         # Store assistant's response in long-term memory
#                         st.session_state.memory_store.store_message(
#                             session_id=st.session_state.session_id,
#                             role="assistant",
#                             content=answer
#                         )

#                     # Handle voice output
#                     if st.session_state.last_input_was_voice:
#                         with st.spinner("Converting response to speech..."):
#                             audio_stream = st.session_state.voice_interface.text_to_speech_stream(answer)
#                             st.session_state.voice_interface.play_audio_stream(audio_stream)

#                 except Exception as e:
#                     error_message = f"An error occurred: {str(e)}"
#                     st.error(error_message)
#                     st.session_state.messages.append({"role": "assistant", "content": error_message})

# if __name__ == "__main__":
#     app()








import streamlit as st
from agents.qa_agent import QAAgent
from database.vectorstore import VectorStore
from agents.content_agent import ContentAgent
from agents.research_crew import ResearchCrew
from voice_interface import VoiceInterface
from memory_store import MemoryStore
from dotenv import load_dotenv
import os
import logging
import uuid
import json

logging.getLogger('root').setLevel(logging.ERROR)

def get_session_id():
    """Ensure consistent session ID across app restarts."""
    # Try to load existing session ID from file
    try:
        if os.path.exists('session.json'):
            with open('session.json', 'r') as f:
                data = json.load(f)
                return data['session_id']
    except:
        pass
    
    # Create new session ID if none exists
    session_id = str(uuid.uuid4())
    try:
        with open('session.json', 'w') as f:
            json.dump({'session_id': session_id}, f)
    except:
        pass
    return session_id

def app():
    st.title("🤖 Chat with Your Learning Assistant")
    load_dotenv()

    # Initialize persistent session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = get_session_id()

    # Initialize memory store if not exists
    if 'memory_store' not in st.session_state:
        st.session_state.memory_store = MemoryStore()

    # Initialize voice interface
    if 'voice_interface' not in st.session_state:
        st.session_state.voice_interface = VoiceInterface()

    # Initialize voice input tracker
    if 'last_input_was_voice' not in st.session_state:
        st.session_state.last_input_was_voice = False

    # Initialize components
    llm_choice = st.sidebar.selectbox(
        "Choose an LLM:", ["Groq API", "HuggingFace API"]
    )

    # Add memory toggle in sidebar
    use_memory = st.sidebar.checkbox("Use Chat Memory", value=True)

    vector_store = VectorStore(llm_choice=llm_choice)
    content_agent = ContentAgent(vector_store)
    qa_agent = QAAgent(
        vector_store,
        llm_choice=llm_choice,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        hf_key=os.getenv("HF_API_KEY"),
    )

    research_crew = ResearchCrew(
        content_agent=content_agent,
        qa_agent=qa_agent,
        vector_store=vector_store
    )

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Load previous messages from memory store
        if use_memory:
            previous_messages = st.session_state.memory_store.get_recent_history(
                st.session_state.session_id, 
                limit=10
            )
            for role, content in previous_messages:
                st.session_state.messages.append({"role": role, "content": content})

    # Create containers
    chat_container = st.container()
    input_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            else:
                st.chat_message("assistant").markdown(message["content"])

    # Add input elements
    with input_container:
        col1, col2 = st.columns([1, 5])
        
        # Text input
        with col2:
            text_prompt = st.chat_input("Type or press 🎤 to speak...")
            if text_prompt:
                prompt = text_prompt
                st.session_state.last_input_was_voice = False

        # Voice input
        with col1:
            if st.button("🎤 Record"):
                with st.spinner("Listening..."):
                    transcribed_text = st.session_state.voice_interface.record_audio()
                    if transcribed_text:
                        prompt = transcribed_text
                        st.session_state.last_input_was_voice = True
                    else:
                        st.error("No speech detected. Please try again.")
                        st.session_state.last_input_was_voice = False
                        prompt = None

        # Process input
        if 'prompt' in locals() and prompt:
            with chat_container:
                # Store user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").markdown(prompt)

                if use_memory:
                    # Store in memory store
                    st.session_state.memory_store.store_message(
                        session_id=st.session_state.session_id,
                        role="user",
                        content=prompt
                    )
                    # Get conversation context
                    conversation_context = st.session_state.memory_store.get_conversation_context(
                        session_id=st.session_state.session_id
                    )
                else:
                    conversation_context = None

                try:
                    # Get response with context
                    qa_response = qa_agent.get_answer(
                        prompt, 
                        history_context=conversation_context if use_memory else None
                    )
                    
                    if qa_response["sources"] and qa_response["sources"] != ["No specific sources"]:
                        answer = qa_response["answer"]
                        sources = qa_response["sources"]
                    else:
                        with st.spinner("Researching your question..."):
                            research_response = research_crew.process_query(prompt)
                            answer = research_response["answer"]
                            sources = research_response["sources"]

                    # Format response
                    sources_display = "\n".join(sources) if sources else "No sources available."
                    full_response = f"{answer}\n\n**Sources:**\n{sources_display}"
                    
                    st.chat_message("assistant").markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    if use_memory:
                        # Store assistant's response
                        st.session_state.memory_store.store_message(
                            session_id=st.session_state.session_id,
                            role="assistant",
                            content=answer
                        )

                    # Handle voice output
                    if st.session_state.last_input_was_voice:
                        with st.spinner("Converting response to speech..."):
                            audio_stream = st.session_state.voice_interface.text_to_speech_stream(answer)
                            st.session_state.voice_interface.play_audio_stream(audio_stream)

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Add button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        if use_memory:
            st.session_state.memory_store.clear_history(st.session_state.session_id)
        st.rerun()

if __name__ == "__main__":
    app()