# from crewai import Agent
# from langchain.prompts import PromptTemplate
# from pydantic import PrivateAttr
# from typing import List, Dict, Optional
# import random

# class CheatSheetAgent(Agent):
#     _vector_store: PrivateAttr
#     _llm: PrivateAttr

#     def __init__(self, vector_store, qa_agent):
#         super().__init__(
#             role="Study Material Summarizer",
#             goal="Create concise, effective one-page summaries of educational content",
#             backstory="""Expert at distilling complex information into clear, 
#                      memorable summaries with key points and concepts.""",
#             allow_delegation=True,
#             llm=qa_agent._llm,
#             verbose=True
#         )
#         self._vector_store = vector_store
#         self._llm = qa_agent._llm

#     def create_cheatsheet(self, topic: str) -> Dict[str, str]:
#         """
#         Create a one-page summary of the given topic.
        
#         Args:
#             topic: The subject to create a cheat sheet for
            
#         Returns:
#             dict: Contains sections of the cheat sheet
#         """
#         # Get relevant content from vector store
#         retriever = self._vector_store.as_retriever()
#         docs = retriever.get_relevant_documents(topic)
        
#         if not docs:
#             return {
#                 "error": "No relevant content found for this topic."
#             }

#         context = "\n".join([doc.page_content for doc in docs])
        
#         cheatsheet_prompt = f"""Create a one-page cheat sheet for the following topic using this context:
        
#         Topic: {topic}
#         Context: {context}
        
#         Structure the cheat sheet with these sections:
#         1. Key Concepts (maximum 5 bullet points)
#         2. Important Definitions (maximum 4)
#         3. Core Principles (maximum 3)
#         4. Quick Reference Examples (maximum 2)
#         5. Common Pitfalls to Avoid (maximum 3)
        
#         Keep each section concise and focused. Format in Markdown."""

#         try:
#             response = self._llm.predict(cheatsheet_prompt)
#             return {
#                 "content": response,
#                 "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
#             }
#         except Exception as e:
#             return {"error": f"Error generating cheat sheet: {str(e)}"}

# class QuizAgent(Agent):
#     _vector_store: PrivateAttr
#     _llm: PrivateAttr

#     def __init__(self, vector_store, qa_agent):
#         super().__init__(
#             role="Educational Assessment Expert",
#             goal="Create engaging and effective quiz questions to test understanding",
#             backstory="""Expert at creating various types of assessment questions 
#                      that effectively test comprehension and knowledge retention.""",
#             allow_delegation=True,
#             llm=qa_agent._llm,
#             verbose=True
#         )
#         self._vector_store = vector_store
#         self._llm = qa_agent._llm

#     def generate_quiz(self, topic: str, num_questions: int = 5, question_type: str = "multiple_choice") -> Dict[str, any]:
#         """
#         Generate a quiz for the given topic.
        
#         Args:
#             topic: The subject to create questions about
#             num_questions: Number of questions to generate
#             question_type: Type of questions ("multiple_choice" or "open_ended")
            
#         Returns:
#             dict: Contains quiz questions and answers
#         """
#         retriever = self._vector_store.as_retriever()
#         docs = retriever.get_relevant_documents(topic)
        
#         if not docs:
#             return {
#                 "error": "No relevant content found for this topic."
#             }

#         context = "\n".join([doc.page_content for doc in docs])
        
#         if question_type == "multiple_choice":
#             prompt_template = f"""Create {num_questions} multiple choice questions based on this content:
            
#             Content: {context}
            
#             For each question:
#             1. Provide the question
#             2. Give 4 possible answers (A, B, C, D)
#             3. Indicate the correct answer
#             4. Include a brief explanation of why it's correct
            
#             Format each question as a JSON-like structure with keys: question, options, correct_answer, explanation"""
            
#         else:  # open_ended
#             prompt_template = f"""Create {num_questions} open-ended questions based on this content:
            
#             Content: {context}
            
#             For each question:
#             1. Provide the question
#             2. Give a model answer
#             3. Include 2-3 key points that a good answer should contain
            
#             Format each question as a JSON-like structure with keys: question, model_answer, key_points"""

#         try:
#             response = self._llm.predict(prompt_template)
#             return {
#                 "questions": response,
#                 "topic": topic,
#                 "type": question_type,
#                 "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
#             }
#         except Exception as e:
#             return {"error": f"Error generating quiz: {str(e)}"}

#     def grade_open_ended_response(self, question: str, model_answer: str, key_points: List[str], student_response: str) -> Dict[str, any]:
#         """
#         Grade an open-ended response against model answer and key points.
        
#         Args:
#             question: The quiz question
#             model_answer: The model answer to compare against
#             key_points: List of key points that should be covered
#             student_response: The student's answer to grade
            
#         Returns:
#             dict: Contains grade and feedback
#         """
#         grading_prompt = f"""Grade this student response against the model answer and key points:
        
#         Question: {question}
#         Model Answer: {model_answer}
#         Key Points Required: {", ".join(key_points)}
        
#         Student Response: {student_response}
        
#         Provide:
#         1. Score (0-100)
#         2. Specific feedback on what was good
#         3. Areas for improvement
#         4. Missing key points"""

#         try:
#             response = self._llm.predict(grading_prompt)
#             return {
#                 "feedback": response
#             }
#         except Exception as e:
#             return {"error": f"Error grading response: {str(e)}"}



from crewai import Agent
from langchain.prompts import PromptTemplate
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
import json

class CheatSheetAgent(Agent):
    _vector_store: PrivateAttr
    _llm: PrivateAttr

    def __init__(self, vector_store, qa_agent):
        super().__init__(
            role="Study Material Summarizer",
            goal="Create concise, effective one-page summaries of educational content",
            backstory="""Expert at distilling complex information into clear, 
                     memorable summaries with key points and concepts.""",
            allow_delegation=True,
            llm=qa_agent._llm,
            verbose=True
        )
        self._vector_store = vector_store
        self._llm = qa_agent._llm

    def create_cheatsheet(self, topic: str) -> Dict[str, str]:
        """
        Create a one-page summary of the given topic.
        """
        retriever = self._vector_store.as_retriever()
        docs = retriever.get_relevant_documents(topic)
        
        if not docs:
            return {
                "error": "No relevant content found for this topic."
            }

        context = "\n".join([doc.page_content for doc in docs])
        
        cheatsheet_prompt = f"""Create a one-page cheat sheet for the following topic using this context:
        
        Topic: {topic}
        Context: {context}
        
        Structure the cheat sheet with these sections:
        1. Key Concepts (maximum 5 bullet points)
        2. Important Definitions (maximum 4)
        3. Core Principles (maximum 3)
        4. Quick Reference Examples (maximum 2)
        5. Common Pitfalls to Avoid (maximum 3)
        
        Keep each section concise and focused. Format in Markdown."""

        try:
            response = self._llm.predict(cheatsheet_prompt)
            return {
                "content": response,
                "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
            }
        except Exception as e:
            return {"error": f"Error generating cheat sheet: {str(e)}"}

class QuizAgent(Agent):
    _vector_store: PrivateAttr
    _llm: PrivateAttr

    def __init__(self, vector_store, qa_agent):
        super().__init__(
            role="Educational Assessment Expert",
            goal="Create engaging and effective quiz questions to test understanding",
            backstory="""Expert at creating various types of assessment questions 
                     that effectively test comprehension and knowledge retention.""",
            allow_delegation=True,
            llm=qa_agent._llm,
            verbose=True
        )
        self._vector_store = vector_store
        self._llm = qa_agent._llm

    def generate_quiz(self, topic: str, num_questions: int = 5, question_type: str = "multiple_choice") -> Dict[str, any]:
        """Generate a quiz with proper formatting for the frontend."""
        retriever = self._vector_store.as_retriever()
        docs = retriever.get_relevant_documents(topic)
        
        if not docs:
            return {"error": "No relevant content found for this topic."}

        context = "\n".join([doc.page_content for doc in docs])
        
        if question_type == "multiple_choice":
            prompt = f"""Generate {num_questions} multiple choice questions about {topic} based on this content: {context}

Your response must be a valid JSON array containing exactly {num_questions} questions.
Do not include any additional text before or after the JSON array.
Each question in the array must follow this exact structure:
[
    {{
        "question": "What is...",
        "options": ["First option", "Second option", "Third option", "Fourth option"],
        "correct_answer": "Second option",
        "explanation": "This is correct because..."
    }}
]"""
        else:
            prompt = f"""Generate {num_questions} open-ended questions about {topic} based on this content: {context}

Your response must be a valid JSON array containing exactly {num_questions} questions.
Do not include any additional text before or after the JSON array.
Each question in the array must follow this exact structure:
[
    {{
        "question": "Explain...",
        "model_answer": "The complete answer...",
        "key_points": ["First key point", "Second key point", "Third key point"]
    }}
]"""

        try:
            response = self._llm.predict(prompt)
            
            # Clean the response
            response = response.strip()
            
            # Try to find JSON array
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                # If no array found, try to find a JSON object instead
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
            if start_idx == -1 or end_idx <= start_idx:
                print(f"Invalid response format: {response}")
                raise ValueError("Could not find valid JSON in response")
                
            json_str = response[start_idx:end_idx]
            
            # Parse the JSON
            questions = json.loads(json_str)
            
            # If we got a dict with a questions key, extract the questions
            if isinstance(questions, dict) and "questions" in questions:
                questions = questions["questions"]
            
            # Ensure we have a list of questions
            if not isinstance(questions, list):
                questions = [questions]
            
            # Validate each question has required fields
            required_fields = ["question", "options", "correct_answer", "explanation"] if question_type == "multiple_choice" else ["question", "model_answer", "key_points"]
            
            for q in questions:
                missing_fields = [field for field in required_fields if field not in q]
                if missing_fields:
                    raise ValueError(f"Question missing required fields: {missing_fields}")
            
            return {
                "topic": topic,
                "type": question_type,
                "questions": questions,
                "total_questions": len(questions)
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Response: {response}")
            raise ValueError(f"Failed to parse quiz response: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Response: {response}")
            raise ValueError(f"Failed to generate quiz: {str(e)}")


    def grade_open_ended_response(self, question_data: Dict[str, Any], student_response: str) -> Dict[str, Any]:
        """Grade an open-ended response with detailed feedback."""
        try:
            grading_prompt = f"""Grade this answer:
Question: {question_data['question']}
Model Answer: {question_data['model_answer']}
Key Points to Check: {', '.join(question_data['key_points'])}
Student Answer: {student_response}

Respond with a JSON object only, following this exact structure:
{{
    "score": 85,
    "feedback": "Your feedback here",
    "key_points_covered": ["Point 1", "Point 2"],
    "missing_points": ["Point 3"],
    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
}}"""

            response = self._llm.predict(grading_prompt)
            response = response.strip()
            
            # Find JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("Could not find valid JSON in response")
                
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"Grading Error: {str(e)}")
            print(f"Response: {response}")
            return {
                "score": 0,
                "feedback": "Error processing response",
                "key_points_covered": [],
                "missing_points": [],
                "improvement_suggestions": ["Unable to grade response due to technical error"]
            }