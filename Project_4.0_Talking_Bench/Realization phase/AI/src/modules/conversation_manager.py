import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import os

from core.llm_handler import LLMHandler
from modules.emotion_analyzer import EmotionAnalyzer
from modules.filler_agent import FillerAgent
from modules.message_templates import MessageTemplates

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.bench_data = None
        self.llm_handler = None
        self.emotion_analyzer = None
        self.filler_agent = None
        self.message_templates = None
        self.current_question_index = 0
        self.processing_threshold = 2.0
        self.conversation_history = []
        self.follow_up_questions = []
        self.waiting_for_response = False
        self.follow_up_count = 0
        self.max_follow_ups = 1
        self.final_follow_ups_complete = False
        self.conversation_analytics = {
            "start_time": datetime.utcnow().isoformat(),
            "exchanges": [],
            "emotional_trajectory": [],
            "topics_covered": []
        }
        self.questions = None

    @classmethod
    async def create(cls):
        instance = cls()
        await instance._initialize()
        return instance

    async def _initialize(self):
        try:
            self.bench_data = await self._fetch_bench_data()
            if not self.bench_data:
                raise ValueError("Kan bench data niet ophalen")
                
            self.llm_handler = LLMHandler(self.bench_data)
            self.emotion_analyzer = EmotionAnalyzer()
            self.filler_agent = FillerAgent(self.bench_data)
            self.message_templates = MessageTemplates()
            self.questions = self.bench_data["location"]["questions"]
            
        except Exception as e:
            logger.error(f"Initialisatie van ConversationManager mislukt: {str(e)}")
            raise

    async def _fetch_bench_data(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(os.getenv('BENCH_API_URL')) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"API verzoek mislukt: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Ophalen van bench data mislukt: {str(e)}")
            return None

    def _is_ending_conversation(self, user_input: str) -> bool:
        normalized_input = user_input.lower().strip()
        
        exit_phrases = [
            'doei', 'tot ziens', 'vaarwel', 'tot kijk',
            'ik moet gaan', 'ik ga weg', 'tijd om te gaan',
            'geen vragen meer', 'dat was alles', 'ik ben klaar',
            'stop met praten', 'beëindig gesprek', 'laten we stoppen',
            'dank je wel en dag', 'bedankt doei',
            'geen verdere vragen', 'heb je nog meer vragen',
            'zijn er nog meer vragen', 'zijn we klaar',
            'kunnen we stoppen', 'kunnen we eindigen', 'dit is genoeg'
        ]
        
        return any(phrase in normalized_input for phrase in exit_phrases)

    def _is_navigation_command(self, user_input: str) -> bool:
        commands = ["volgende vraag", "vorige vraag", "ga terug", 
                   "doorgaan", "overslaan"]
        return any(cmd in user_input.lower() for cmd in commands)

    async def _handle_navigation(self, user_input: str) -> Dict[str, Any]:
        if "volgende" in user_input.lower():
            if self.current_question_index < len(self.questions) - 1:
                self.current_question_index += 1
                self.follow_up_questions = []
                return {
                    "response": self.questions[self.current_question_index]["content"],
                    "metadata": {"navigation": "next"}
                }
            return {
                "response": "We zijn aan het einde gekomen van mijn vragen. Wil je nog ergens anders over praten?",
                "metadata": {"navigation": "end"}
            }
        
        if "vorige" in user_input.lower() or "terug" in user_input.lower():
            if self.current_question_index > 0:
                self.current_question_index -= 1
                self.follow_up_questions = []
                return {
                    "response": self.questions[self.current_question_index]["content"],
                    "metadata": {"navigation": "previous"}
                }
            return {
                "response": "We zijn aan het begin. Wil je doorgaan?",
                "metadata": {"navigation": "start"}
            }

    async def process_conversation(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            if self._is_ending_conversation(user_input):
                return await self.end_conversation()

            if self._is_navigation_command(user_input):
                return await self._handle_navigation(user_input)

            current_question = self.questions[self.current_question_index]
            current_question = {
                'id': current_question.get('id', ''),
                'content': current_question.get('content', ''),
                'context': self.bench_data['location'].get('context', ''),
                'locationId': current_question.get('locationId', '')
            }

            filler = None
            if time.time() - start_time > self.processing_threshold:
                filler = self.filler_agent.get_filler_content()

            is_transitioning = self.follow_up_count >= self.max_follow_ups

            response = await self.llm_handler.generate_response(
                user_input=user_input,
                current_question=current_question,
                conversation_history=self.conversation_history,
                previous_follow_ups=self.follow_up_questions,
                is_transitioning=is_transitioning
            )

            if not self.waiting_for_response:
                self.follow_up_count += 1
                
                if self.follow_up_count > self.max_follow_ups:
                    if self.current_question_index < len(self.questions) - 1:
                        response_text = response['response'].split('\n')[0]
                        self.current_question_index += 1
                        next_question = self.questions[self.current_question_index]["content"]
                        response['response'] = f"{response_text}\n\nLaten we verdergaan. {next_question}"
                        self.follow_up_count = 0
                        self.follow_up_questions = []
                        self.waiting_for_response = False
                    else:
                        return await self.end_conversation()
                elif response.get('follow_up_questions'):
                    follow_up = response['follow_up_questions'][0]
                    response['response'] = response['response'].split('\n')[0] + f"\n\n{follow_up}"
                    self.follow_up_questions = [follow_up]
                    self.waiting_for_response = True
            else:
                self.waiting_for_response = False

            emotional_insights = await self.emotion_analyzer.get_conversation_insights(
                self.conversation_history[-5:]
            )

            self.conversation_analytics["exchanges"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "response": response["response"],
                "emotional_analysis": emotional_insights,
                "question_id": current_question["id"]
            })
            
            self.conversation_analytics["emotional_trajectory"].append(emotional_insights["emotional_summary"])
            
            if current_question["context"] not in self.conversation_analytics["topics_covered"]:
                self.conversation_analytics["topics_covered"].append(current_question["context"])

            self.conversation_history.append({
                'user': user_input,
                'response': response['response'],
                'timestamp': datetime.utcnow().isoformat(),
                'context': {
                    'question_id': current_question['id'],
                    'emotional_context': response['emotional_context'],
                    'error': response.get('metadata', {}).get('error', '')
                }
            })

            final_response = self._build_response(response, filler)

            return {
                "response": final_response,
                "follow_ups": [] if self.waiting_for_response else self.follow_up_questions,
                "emotional_analysis": emotional_insights,
                "metadata": {
                    "question_index": self.current_question_index,
                    "processing_time": time.time() - start_time,
                    "confidence": response.get('confidence', 0.5),
                    "needs_clarification": response.get('needs_clarification', False),
                    "time_awareness": response.get('time_awareness', {}),
                    "error": response.get('metadata', {}).get('error', ''),
                    "waiting_for_response": self.waiting_for_response,
                    "follow_up_count": self.follow_up_count
                }
            }

        except Exception as e:
            logger.error(f"Verwerking van gesprek mislukt: {str(e)}")
            return {
                "response": "Sorry, er ging iets mis bij het verwerken. Kunnen we het opnieuw proberen?",
                "error": str(e)
            }

    def get_welcome_message(self) -> str:
        return self.message_templates.get_welcome_message(self.get_current_question())

    def get_current_question(self) -> str:
        if not self.questions:
            return "Geen vragen beschikbaar."
        return self.questions[self.current_question_index]["content"]

    def get_conversation_summary(self) -> Dict[str, Any]:
        if not self.conversation_history:
            return {"summary": "Geen gespreksgeschiedenis beschikbaar."}
            
        return {
            "exchanges": len(self.conversation_history),
            "current_topic": self.questions[self.current_question_index]["context"],
            "follow_ups_pending": len(self.follow_up_questions),
            "last_exchange": {
                "timestamp": self.conversation_history[-1]["timestamp"],
                "emotional_context": self.conversation_history[-1]["context"]["emotional_context"]
            }
        }

    def get_conversation_analytics(self) -> Dict[str, Any]:
        return {
            **self.conversation_analytics,
            "duration": (datetime.utcnow() - datetime.fromisoformat(self.conversation_analytics["start_time"])).total_seconds(),
            "questions_covered": self.current_question_index + 1,
            "total_exchanges": len(self.conversation_analytics["exchanges"])
        }

    async def end_conversation(self) -> Dict[str, Any]:
        analytics = self.get_conversation_analytics()
        farewell = self.get_farewell_message()
        
        return {
            "response": farewell,
            "analytics": analytics,
            "metadata": {
                "conversation_ended": True,
                "duration": analytics["duration"],
                "exchanges": analytics["total_exchanges"]
            }
        }

    def get_farewell_message(self) -> str:
        if not self.conversation_analytics["start_time"]:
            return "Bedankt voor het gesprek! Nog een fijne dag!"
            
        start_time = datetime.fromisoformat(self.conversation_analytics["start_time"])
        duration = (datetime.utcnow() - start_time).total_seconds()
        return self.message_templates.get_farewell_message(duration)

    def _build_response(self, result: Dict[str, Any], filler: Optional[Dict], transition_text: str = "") -> str:
        response = result.get("response", "")
        if filler:
            response = f"{filler['thinking']} {response}"
        return f"{response}{transition_text}"
