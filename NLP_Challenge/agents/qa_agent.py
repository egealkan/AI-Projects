from crewai import Agent
from langchain.chains import RetrievalQA
from helpers.groq_api import GroqAPI
from helpers.huggingface_api import HuggingFaceAPI
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from pydantic import PrivateAttr
from typing import Optional, Dict, List

class QAAgent(Agent):
    _vector_store: PrivateAttr
    _llm: PrivateAttr
    _use_chain: PrivateAttr
    _retriever: PrivateAttr
    _qa_chain: PrivateAttr

    def __init__(self, vector_store, llm_choice, groq_api_key=None, hf_key=None):
        # Initialize LLM based on user choice
        if llm_choice == "Groq API":
            llm = GroqAPI(api_key=groq_api_key, model_name="mixtral-8x7b-32768")
            use_chain = True
        elif llm_choice == "HuggingFace API":
            llm = HuggingFaceAPI(api_key=hf_key, model_name="tiiuae/falcon-7b-instruct")
            use_chain = False
        else:
            raise ValueError(f"Unsupported LLM choice: {llm_choice}")

        # Initialize the Agent with CrewAI parameters
        super().__init__(
            role="Question Answering Expert",
            goal="Provide accurate and detailed answers based on available context",
            backstory="Expert at analyzing context and providing comprehensive answers",
            allow_delegation=True,
            llm=llm,
            verbose=True
        )
        
        # Store private attributes
        self._vector_store = vector_store
        self._llm = llm
        self._use_chain = use_chain
        self._retriever = vector_store.as_retriever()
        
        # Initialize QA chain if using chain
        if use_chain:
            if not isinstance(self._retriever, BaseRetriever):
                raise ValueError("Retriever must be an instance of BaseRetriever")
                
            # Create custom prompt template that includes historical context
            qa_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            
            Context: {context}
            Question: {question}
            
            Please provide a detailed and accurate answer based on the context above."""

            PROMPT = PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            )
            
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self._retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        else:
            self._qa_chain = None

    def retrieve_content(self, question: str) -> tuple[Optional[str], List[str]]:
        """
        Retrieve relevant content from the vector store.
        
        Args:
            question: The user's question
            
        Returns:
            tuple: (context string or None, list of sources)
        """
        try:
            retrieved_docs = self._retriever.get_relevant_documents(question)
            if not retrieved_docs:
                return None, []
            
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Check context relevance
            relevance_prompt = f"""Is this context relevant to answering the question?
    Context: {context}
    Question: {question}
    Answer with just yes or no:"""
            
            relevance_check = self._llm.predict(relevance_prompt).lower().strip()
            if 'yes' not in relevance_check:
                return None, []
                
            sources = [doc.metadata.get("source", "Unknown source") for doc in retrieved_docs]
            return context, sources
        except Exception as e:
            print(f"Error retrieving content: {str(e)}")
            return None, []

    def web_search(self, question: str) -> Optional[str]:
        """
        Perform a web search for the question.
        
        Args:
            question: The search query
            
        Returns:
            str or None: Combined search results or None if no results found
        """
        try:
            # Perform Google search
            search_url = f"https://www.google.com/search?q={requests.utils.quote(question)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = []
                
                # Look for featured snippets
                featured_snippet = soup.find('div', {'class': 'ILfuVd'})
                if featured_snippet:
                    search_results.append(featured_snippet.get_text())
                
                # Look for regular search results
                results = soup.find_all('div', {'class': 'g'})
                for result in results[:3]:
                    title_elem = result.find('h3')
                    snippet_elem = result.find('div', {'class': 'VwiC3b'})
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text()
                        snippet = snippet_elem.get_text()
                        search_results.append(f"{title}: {snippet}")
                
                if search_results:
                    return "\n\n".join(search_results)
            
            # Fallback to news search if regular search fails
            news_url = f"https://news.google.com/rss/search?q={requests.utils.quote(question)}"
            news_response = requests.get(news_url)
            if news_response.status_code == 200:
                news_soup = BeautifulSoup(news_response.text, 'xml')
                items = news_soup.find_all('item')
                if items:
                    results = []
                    for item in items[:3]:
                        title = item.title.text if item.title else ""
                        desc = item.description.text if item.description else ""
                        results.append(f"{title}\n{desc}")
                    return "\n\n".join(results)
            
            return None

        except Exception as e:
            print(f"Web search error: {str(e)}")
            return None

    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            question: The user's question
            context: Optional context to include in the prompt
            
        Returns:
            str: The generated response
        """
        if self._use_chain:  # For Groq API
            if context:
                prompt = f"""Using this context: {context}

Answer this question: {question}

Provide a clear, factual answer based on the context provided."""
            else:
                prompt = f"""Question: {question}
Answer: Provide a clear, factual answer."""
        else:  # For HuggingFace API - simplified prompt
            prompt = f"Define {question} in one sentence:"

        response = self._llm.predict(prompt)
        
        # Clean up HuggingFace responses
        if not self._use_chain:
            response = response.split('\n')[1].strip()
        
        return response

    def get_answer(self, question: str, history_context: Optional[str] = None) -> Dict[str, any]:
        """
        Get an answer to the question using available resources.
        
        Args:
            question: The user's question
            history_context: Optional historical context from previous conversations
            
        Returns:
            dict: Contains 'answer' and 'sources' keys
        """
        try:
            # Try to retrieve relevant content from uploaded documents
            context, sources = self.retrieve_content(question)
            
            # If we have relevant document context, use it
            if context:
                if self._use_chain and self._qa_chain:
                    response = self._qa_chain({
                        "query": question,
                        "context": context
                    })
                    answer = response.get("result", "No answer found.")
                    sources = [doc.metadata.get("source", "Unknown source") for doc in response.get("source_documents", [])]
                else:
                    raw_answer = self.generate_response(question, context)
                    answer_lines = [line for line in raw_answer.split('\n') if line.strip() and 
                                'objective' not in line.lower() and 
                                'provide' not in line.lower()]
                    answer = ' '.join(answer_lines).strip()
            else:
                # No relevant document context found, try web search
                web_results = self.web_search(question)
                
                if web_results:
                    web_prompt = f"""Based on this information:
    {web_results}

    {"Historical context: " + history_context if history_context else ""}

    Please answer the question: {question}

    Provide only the factual answer without any disclaimers about the knowledge cutoff date."""
                    
                    answer = self._llm.predict(web_prompt)
                    sources = ["Recent Web Search"]
                else:
                    # Handle questions about current events or general knowledge
                    if any(keyword in question.lower() for keyword in ["latest", "recent", "current", "2022", "2023", "2024"]):
                        answer = "I apologize, but I'm unable to access current information for this question. Please check a recent news source or official website for the most up-to-date information."
                        sources = ["Unable to access current information"]
                    else:
                        # Use LLM's general knowledge
                        raw_answer = self.generate_response(question)
                        if not self._use_chain:
                            answer_lines = [line for line in raw_answer.split('\n') if line.strip() and 
                                        'objective' not in line.lower() and 
                                        'provide' not in line.lower()]
                            answer = ' '.join(answer_lines).strip()
                        else:
                            answer = raw_answer
                        sources = ["AI Knowledge Base"]

            return {
                "answer": answer,
                "sources": sources
            }
        
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": ["Error occurred during processing"],
            }

