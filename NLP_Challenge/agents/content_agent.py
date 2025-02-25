from crewai import Agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from pydantic import PrivateAttr
import requests
from bs4 import BeautifulSoup
from typing import List, Optional


class ContentAgent(Agent):
    _vector_store: PrivateAttr
    _text_splitter: PrivateAttr

    def __init__(self, vector_store):
        super().__init__(
            role="Content Processing Expert",
            goal="Process and index various educational content effectively",
            backstory="""Expert at analyzing, organizing, and processing diverse educational 
                     materials with a deep understanding of content structure and metadata.""",
            allow_delegation=True,
        )
        self._vector_store = vector_store
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_chunks(self, documents: List[Document], source_type: str) -> str:
        """
        Split and store document chunks with metadata.
        
        Args:
            documents: List of Document objects to process
            source_type: Type of content being processed (PDF, PowerPoint, etc.)
            
        Returns:
            str: Status message about processing
        """
        try:
            chunks = self._text_splitter.split_documents(documents)
            for chunk in chunks:
                chunk.metadata["source_type"] = source_type
            self._vector_store.add_documents(chunks)
            return f"Successfully processed and indexed {len(chunks)} chunks from {source_type}."
        except Exception as e:
            return f"Error processing {source_type} content: {str(e)}"


class PDFIngestionAgent(Agent):
    _content_agent: PrivateAttr

    def __init__(self, content_agent: ContentAgent):
        super().__init__(
            role="PDF Content Expert",
            goal="Extract and process content from PDF files efficiently",
            backstory="""Specialized in handling PDF documents with expertise in 
                     extracting structured and unstructured content.""",
            allow_delegation=True,
        )
        self._content_agent = content_agent

    def ingest_pdf(self, file_path: str) -> str:
        """
        Process and index PDF documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Status message about processing
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self._content_agent.process_chunks(documents, "PDF")
        except Exception as e:
            return f"Error processing PDF file: {str(e)}"


class PowerPointAgent(Agent):
    _content_agent: PrivateAttr

    def __init__(self, content_agent: ContentAgent):
        super().__init__(
            role="PowerPoint Content Expert",
            goal="Extract and process content from PowerPoint presentations effectively",
            backstory="""Specialized in analyzing and processing PowerPoint presentations, 
                     including slides, notes, and embedded content.""",
            allow_delegation=True,
        )
        self._content_agent = content_agent

    def ingest_powerpoint(self, file_path: str) -> str:
        """
        Process and index PowerPoint presentations.
        
        Args:
            file_path: Path to the PowerPoint file
            
        Returns:
            str: Status message about processing
        """
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            documents = loader.load()
            return self._content_agent.process_chunks(documents, "PowerPoint")
        except Exception as e:
            return f"Error processing PowerPoint file: {str(e)}"


class YouTubeAgent(Agent):
    _content_agent: PrivateAttr

    def __init__(self, content_agent: ContentAgent):
        super().__init__(
            role="YouTube Content Expert",
            goal="Extract and process content from YouTube videos efficiently",
            backstory="""Specialized in processing YouTube video transcripts and 
                     converting them into searchable content.""",
            allow_delegation=True,
        )
        self._content_agent = content_agent

    def ingest_youtube(self, video_url: str) -> str:
        """
        Fetch and process YouTube video transcripts.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            str: Status message about processing
        """
        try:
            video_id = video_url.split("v=")[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = "\n".join([item['text'] for item in transcript])
            document = Document(
                page_content=transcript_text,
                metadata={"source": video_url, "type": "video_transcript"}
            )
            return self._content_agent.process_chunks([document], "YouTube")
        except Exception as e:
            return f"Error processing YouTube video: {str(e)}"


class WebSearchAgent(Agent):
    _content_agent: PrivateAttr

    def __init__(self, content_agent: ContentAgent):
        super().__init__(
            role="Web Search Expert",
            goal="Perform web searches and process results effectively",
            backstory="""Specialized in web content retrieval and processing, 
                     with expertise in extracting relevant information from search results.""",
            allow_delegation=True,
        )
        self._content_agent = content_agent

    def ingest_web_search(self, query: str) -> str:
        """
        Perform web search and process results.
        
        Args:
            query: Search query string
            
        Returns:
            str: Status message about processing
        """
        try:
            search_url = f"https://www.google.com/search?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            response = requests.get(search_url, headers=headers)
            if response.status_code != 200:
                return f"Failed to fetch search results. Status code: {response.status_code}"

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("div", class_="tF2Cxc")

            if not results:
                return "No search results found. The page structure may have changed."

            documents = []
            for idx, result in enumerate(results[:5]):  # Limit to top 5 results
                title = result.find("h3").text if result.find("h3") else "No Title"
                snippet = result.find("span", class_="aCOpRe").text if result.find("span", class_="aCOpRe") else "No Snippet"
                link = result.find("a")["href"] if result.find("a") else None

                if title and snippet and link:
                    documents.append(
                        Document(
                            page_content=f"{title}\n{snippet}",
                            metadata={
                                "source": link,
                                "title": title,
                                "rank": idx + 1
                            }
                        )
                    )

            if not documents:
                return "No valid search results found for the query."

            return self._content_agent.process_chunks(documents, "Web Search")
        except Exception as e:
            return f"Error performing web search: {str(e)}"


class MultiSourceIngestionAgent(Agent):
    _content_agent: PrivateAttr
    _pdf_agent: PrivateAttr
    _powerpoint_agent: PrivateAttr
    _youtube_agent: PrivateAttr
    _web_agent: PrivateAttr

    def __init__(self, content_agent: ContentAgent):
        super().__init__(
            role="Multi-Source Content Expert",
            goal="Coordinate content ingestion from multiple sources",
            backstory="""Master coordinator specializing in managing and processing 
                     content from various sources while maintaining consistency.""",
            allow_delegation=True,
        )
        self._content_agent = content_agent
        self._pdf_agent = PDFIngestionAgent(content_agent)
        self._powerpoint_agent = PowerPointAgent(content_agent)
        self._youtube_agent = YouTubeAgent(content_agent)
        self._web_agent = WebSearchAgent(content_agent)

    def process_multiple_sources(
        self,
        pdf_paths: Optional[List[str]] = None,
        pptx_paths: Optional[List[str]] = None,
        youtube_urls: Optional[List[str]] = None,
        search_queries: Optional[List[str]] = None
    ) -> dict:
        """
        Process content from multiple sources in parallel.
        
        Args:
            pdf_paths: List of PDF file paths
            pptx_paths: List of PowerPoint file paths
            youtube_urls: List of YouTube video URLs
            search_queries: List of web search queries
            
        Returns:
            dict: Processing results for each source type
        """
        results = {
            "pdf": [],
            "powerpoint": [],
            "youtube": [],
            "web_search": []
        }

        if pdf_paths:
            for path in pdf_paths:
                results["pdf"].append(self._pdf_agent.ingest_pdf(path))

        if pptx_paths:
            for path in pptx_paths:
                results["powerpoint"].append(self._powerpoint_agent.ingest_powerpoint(path))

        if youtube_urls:
            for url in youtube_urls:
                results["youtube"].append(self._youtube_agent.ingest_youtube(url))

        if search_queries:
            for query in search_queries:
                results["web_search"].append(self._web_agent.ingest_web_search(query))

        return results













