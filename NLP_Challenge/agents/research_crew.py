from crewai import Crew, Process, Task
from typing import Optional

class ResearchCrew:
    def __init__(self, content_agent, qa_agent, vector_store):
        self.content_agent = content_agent
        self.qa_agent = qa_agent
        self.vector_store = vector_store

    def create_research_task(self, query: str) -> Task:
        """Create a research task for content search and analysis."""
        return Task(
            description=f"Research and analyze information related to: {query}",
            agent=self.content_agent,
            expected_output="Researched information and analysis"  # Added back
        )

    def create_qa_task(self, query: str, context: Optional[str] = None) -> Task:
        """Create a QA task for generating answers."""
        task_description = (
            f"Answer the following question: {query}\n"
            f"Context: {context if context else 'Use available knowledge and web search if necessary.'}"
        )
        return Task(
            description=task_description,
            agent=self.qa_agent,
            expected_output="Answer with supporting information"  # Added back
        )

    def process_query(self, query: str) -> dict:
        """Process a query using the research crew workflow."""
        try:
            # Create tasks
            research_task = self.create_research_task(query)
            qa_task = self.create_qa_task(query)

            # Create crew
            crew = Crew(
                agents=[self.content_agent, self.qa_agent],
                tasks=[research_task, qa_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute crew workflow
            result = crew.kickoff()
            return {
                "answer": str(result),
                "sources": ["Research Crew Process"]
            }
        except Exception as e:
            return {
                "answer": f"Error in research process: {str(e)}",
                "sources": ["Error occurred during processing"]
            }

    def process_direct_qa(self, query: str) -> dict:
        """Handle direct QA without research when context is available."""
        try:
            qa_task = self.create_qa_task(query)
            crew = Crew(
                agents=[self.qa_agent],
                tasks=[qa_task],
                process=Process.sequential,
                verbose=True
            )
            result = crew.kickoff()
            return {
                "answer": str(result),
                "sources": ["Direct QA response"]
            }
        except Exception as e:
            return {
                "answer": f"Error in QA process: {str(e)}",
                "sources": ["Error occurred during processing"]
            }