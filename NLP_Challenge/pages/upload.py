# import streamlit as st
# from agents.content_agent import ContentAgent
# from langchain.document_loaders import PyPDFLoader
# from database.vectorstore import VectorStore
# from dotenv import load_dotenv
# import os
# import tempfile  # For temporary file handling

# def app():
#     st.title("📤 Upload Learning Materials")
#     load_dotenv()

#     # Initialize VectorStore and ContentAgent
#     vector_store = VectorStore(
#         llm_choice="Groq API",  # Use any embeddings for indexing
#     )
#     content_agent = ContentAgent(vector_store)

#     # File upload section
#     uploaded_file = st.file_uploader("Upload your learning material (PDF only):", type=["pdf"])

#     if uploaded_file:
#         # Save the uploaded file to a temporary location
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_file_path = tmp_file.name  # Get the temporary file path

#         with st.spinner("Processing uploaded file..."):
#             # Process the temporary file path using ContentAgent
#             result_message = content_agent.process_document(tmp_file_path)  # Pass the file path
#             st.success(result_message)

#         # Optionally delete the temporary file (or keep for debugging)
#         os.remove(tmp_file_path)






# import streamlit as st
# from agents.content_agent import (
#     ContentAgent,
#     PDFIngestionAgent,
#     YouTubeAgent,
#     PowerPointAgent,
#     WebSearchAgent,
# )
# from database.vectorstore import VectorStore
# from dotenv import load_dotenv
# import os
# import tempfile


# def app():
#     st.title("📤 Upload Learning Materials")
#     load_dotenv()

#     # Initialize VectorStore and agents
#     vector_store = VectorStore(llm_choice="Groq API")
#     content_agent = ContentAgent(vector_store)
#     pdf_agent = PDFIngestionAgent(content_agent)
#     youtube_agent = YouTubeAgent(content_agent)
#     powerpoint_agent = PowerPointAgent(content_agent)
#     web_search_agent = WebSearchAgent(content_agent)

#     # Upload section
#     upload_type = st.selectbox(
#         "Choose the type of content to upload:",
#         ["PDF", "PowerPoint", "YouTube Link", "Web Search Query"],
#     )

#     if upload_type == "PDF":
#         uploaded_file = st.file_uploader("Upload your PDF file:", type=["pdf"])
#         if uploaded_file:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 tmp_file_path = tmp_file.name

#             with st.spinner("Processing your PDF..."):
#                 result = pdf_agent.ingest_pdf(tmp_file_path)
#                 st.success(result)
#             os.remove(tmp_file_path)

#     elif upload_type == "PowerPoint":
#         uploaded_file = st.file_uploader("Upload your PowerPoint file:", type=["pptx"])
#         if uploaded_file:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 tmp_file_path = tmp_file.name

#             with st.spinner("Processing your PowerPoint..."):
#                 result = powerpoint_agent.ingest_powerpoint(tmp_file_path)
#                 st.success(result)
#             os.remove(tmp_file_path)

#     elif upload_type == "YouTube Link":
#         youtube_link = st.text_input("Enter YouTube video URL:")
#         if st.button("Process YouTube Video"):
#             with st.spinner("Processing YouTube transcript..."):
#                 result = youtube_agent.ingest_youtube(youtube_link)
#                 st.success(result)

#     elif upload_type == "Web Search Query":
#         search_query = st.text_input("Enter your search query:")
#         if st.button("Perform Web Search"):
#             with st.spinner("Performing web search..."):
#                 result = web_search_agent.ingest_web_search(search_query)
#                 st.success(result)






import streamlit as st
from agents.content_agent import ContentAgent, PDFIngestionAgent, PowerPointAgent, YouTubeAgent
from database.vectorstore import VectorStore
from dotenv import load_dotenv
import os
import tempfile


def app():
    st.title("📤 Upload Learning Materials")
    load_dotenv()

    # Initialize VectorStore and agents
    vector_store = VectorStore(llm_choice="Groq API")
    content_agent = ContentAgent(vector_store)
    pdf_agent = PDFIngestionAgent(content_agent)
    youtube_agent = YouTubeAgent(content_agent)
    powerpoint_agent = PowerPointAgent(content_agent)

    # Upload section
    upload_type = st.selectbox(
        "Choose the type of content to upload:",
        ["PDF", "PowerPoint", "YouTube Link"]
    )

    if upload_type == "PDF":
        uploaded_file = st.file_uploader("Upload your PDF file:", type=["pdf"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing your PDF..."):
                result = pdf_agent.ingest_pdf(tmp_file_path)
                st.success(result)
            os.remove(tmp_file_path)

    elif upload_type == "PowerPoint":
        uploaded_file = st.file_uploader("Upload your PowerPoint file:", type=["pptx"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing your PowerPoint..."):
                result = powerpoint_agent.ingest_powerpoint(tmp_file_path)
                st.success(result)
            os.remove(tmp_file_path)

    elif upload_type == "YouTube Link":
        youtube_link = st.text_input("Enter YouTube video URL:")
        if st.button("Process YouTube Video"):
            with st.spinner("Processing YouTube transcript..."):
                result = youtube_agent.ingest_youtube(youtube_link)
                st.success(result)
