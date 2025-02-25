import google.generativeai as genai
import json
import logging
from datetime import datetime
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, bench_data):
        self._setup_llm()
        self.bench_data = bench_data
        
    def _setup_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY niet gevonden")
            
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self._config = {
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1024,
        }

    def _is_repetitive_response(self, response: str, conversation_history: list) -> bool:
        if not conversation_history:
            return False
        last_responses = [entry.get('response', '') for entry in conversation_history[-2:]]
        return any(response.strip() == prev.strip() for prev in last_responses)

    def _build_conversation_prompt(self, user_input: str, current_question: Dict, conversation_history: list, follow_ups: list = None, is_transitioning: bool = False) -> str:
        conversation_history_formatted = self._format_history(conversation_history)
        follow_ups_formatted = json.dumps(follow_ups) if follow_ups else 'Geen'
        
        uncertain_phrases = {"ik weet het niet", "weet niet", "geen idee"}
        uncertainty_instructions = ""
        if user_input.strip().lower() in uncertain_phrases:
            uncertainty_instructions = """
            SPECIAAL GEVAL: De gebruiker lijkt onzeker.
            - Bevestig dat het prima is om niet alle antwoorden te hebben.
            - Toon begrip en ga verder.
            """
        
        prompt = f'''Je bent een intelligente, empathische slimme bank. Je naam is Frank.

    PERSOONLIJKHEIDSKENMERKEN:
    1. Emotionele Intelligentie:
    - Leest subtiele emotionele ondertonen in spraak
    - Gebruikt zachte humor waar gepast
    - Behoudt professionele grenzen terwijl je vriendelijk blijft

    2. Observationele Wijsheid:
    - Verbindt huidige gesprekken met eerdere observaties
    - Maak GEEN verzonnen verhalen over zaken waarvan je NOG NIET weet over de plek of de mensen die daar wonen

    3. Gespreksstijl:
    - Spreekt natuurlijk zoals een mens zou doen
    - Beantwoord altijd eerst directe vragen, ongeacht het onderwerp
    - Deel je kennis over elk onderwerp vanuit je unieke bankperspectief
    - Laat onderwerpen natuurlijk evolueren

    HUIDIGE CONTEXT:
    Locatie: {self.bench_data['location']['name']}
    Locatie Context: {self.bench_data['location']['context']}
    Huidige Vraag: {current_question['content']}
    Vorige Vervolgvragen: {follow_ups_formatted}
    Gebruikersinvoer: {user_input}

    Gespreksgeschiedenis:
    {conversation_history_formatted}

    {uncertainty_instructions}'''

        
        if is_transitioning:
            prompt += """
            BELANGRIJK: Dit is het laatste antwoord voordat we naar een nieuw onderwerp gaan.
            - Geef ALLEEN een korte, reflecterende uitspraak over wat er zojuist is besproken zonder vragen.
            - Genereer GEEN vervolgvraag.
            - Neem GEEN vragen op in je antwoord.
            - Houd het antwoord beknopt.
            """
        else:
            prompt += """
            ANTWOORDREGELS:
            - Genereer precies ÉÉN vervolgvraag per antwoord.
            - Als je meer dan één vervolgvraag genereert, is het antwoord ongeldig.
            - Houd antwoorden gefocust en natuurlijk.
            - Elke vervolgvraag moet uniek zijn en niet eerder gesteld in het gesprek.
            - Herhaal NIET wat de gebruiker heeft gezegd in het vervolgantwoord.
            - Eindig het antwoord NIET enkel met een uitspraak en stel geen relevante vraag.
            - Laat ruimte over voor de gebruiker om het gesprek voort te zetten.

            Je antwoord MOET dit format volgen:
            1. Hoofdantwoord zonder vragen (1-2 zinnen)
            2. ALLEEN een vervolgvraag na het hoofdantwoord
            """

        prompt += """
            INTERACTIERICHTLIJNEN:

            1. Natuurlijke Gespreksflow:
            - Behandel altijd eerst directe vragen, ongeacht het onderwerp.
            - Keer na het beantwoorden van vragen soepel terug naar het hoofdgesprek.
            - Bouw een oprechte connectie op in plaats van te ondervragen.
            - Spreek natuurlijk zoals een mens.
            - Gebruik gespreksmarkeringen ("Ik herinner me...", "Dat doet me denken aan...").

            2. Generatie van Vervolgvragen:
            - Maak één enkele contextuele vraag.
            - Vermijd generieke of herhalende vragen.
            - Bouw voort op gedeelde specifieke details.

            3. Contextbeheer:
            - Verwijs op natuurlijke wijze naar eerdere gesprekspunten.
            - Verbind huidige onderwerpen met lokale kennis.
            - Onthoud belangrijke details van eerdere delen van het gesprek.

            VEREIST OUTPUTFORMAAT:
            {
                "response": "Je natuurlijke gespreksantwoord dat de persoonlijkheid van de bank belichaamt",
                "follow_up_questions": ["ALLEEN één sterk contextuele vervolgvraag"],
                "needs_clarification": false,
                "emotional_context": {
                    "detected_emotion": "geïdentificeerde emotie",
                    "response_tone": "gekozen toon",
                    "engagement_level": "hoog/middel/laag"
                },
                "confidence": 0.9,
                "should_change_topic": false,
                "time_awareness": {
                    "references_time_of_day": false,
                    "references_weather": false,
                    "references_season": false
                }
            }"""

        return prompt

    def _format_history(self, history: list) -> str:
        formatted = []
        for entry in history[-5:]:
            formatted.append(f"Gebruiker: {entry.get('user', '')}")
            formatted.append(f"Bank: {entry.get('response', '')}")
        return "\n".join(formatted)

    def _get_fallback_response(self, error_msg: str = "") -> Dict[str, Any]:
        return {
            "response": "Sorry, ik heb moeite met het verwerken hiervan. Kunnen we het opnieuw proberen?",
            "follow_up_questions": [],
            "needs_clarification": True,
            "emotional_context": {
                "detected_emotion": "neutraal",
                "response_tone": "verontschuldigend",
                "engagement_level": "middel"
            },
            "confidence": 0.0,
            "should_change_topic": False,
            "time_awareness": {
                "references_time_of_day": False,
                "references_weather": False,
                "references_season": False
            },
            "metadata": {
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _check_and_incorporate_context(self, current_response: Dict, conversation_history: list) -> Dict:
        last_message = conversation_history[-1] if conversation_history else None
        if last_message and 'user' in last_message:
            user_input = last_message['user']
            response_text = current_response.get('response', '')
            
            if not any(word in response_text.lower() for word in user_input.lower().split()):
                response_text = f"Over {user_input}, {response_text}"
                current_response['response'] = response_text
                
        return current_response

    async def generate_response(
        self, 
        user_input: str, 
        current_question: Dict, 
        conversation_history: list, 
        previous_follow_ups: list = None,
        is_transitioning: bool = False
    ) -> Dict[str, Any]:
        try:
            logger.info(f"LLM invoer - Huidige vraag: {current_question}")
            logger.info(f"LLM invoer - Vorige vervolgvragen: {previous_follow_ups}")
            
            prompt = self._build_conversation_prompt(
                user_input,
                current_question,
                conversation_history,
                previous_follow_ups,
                is_transitioning
            )
            
            response = self._model.generate_content(prompt, generation_config=self._config)
            
            try:
                clean_text = response.text.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:-3]
                result = json.loads(clean_text)
                
                if result.get("follow_up_questions") and len(result["follow_up_questions"]) > 1:
                    logger.warning("LLM genereerde meerdere vervolgvragen. Beperkt tot één.")
                    result["follow_up_questions"] = [result["follow_up_questions"][0]]
                
                if self._is_repetitive_response(result.get("response", ""), conversation_history):
                    if previous_follow_ups and len(previous_follow_ups) > 0:
                        last_follow_up = previous_follow_ups[-1]
                        new_prompt = self._build_conversation_prompt(
                            last_follow_up,
                            current_question,
                            conversation_history,
                            [],
                            is_transitioning
                        )
                        new_response = self._model.generate_content(new_prompt, generation_config=self._config)
                        result = json.loads(new_response.text.strip())
                
            except json.JSONDecodeError as e:
                logger.error(f"Ongeldige JSON response: {str(e)}: {response.text}")
                return self._get_fallback_response(f"JSON verwerkingsfout: {str(e)}")
            
            return {
                "response": result.get("response", "Sorry, ik moet even mijn gedachten verzamelen."),
                "follow_up_questions": result.get("follow_up_questions", []),
                "needs_clarification": result.get("needs_clarification", False),
                "emotional_context": result.get("emotional_context", {
                    "detected_emotion": "neutraal",
                    "response_tone": "neutraal",
                    "engagement_level": "middel"
                }),
                "confidence": result.get("confidence", 0.5),
                "should_change_topic": result.get("should_change_topic", False),
                "time_awareness": result.get("time_awareness", {}),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "prompt_tokens": len(prompt),
                    "question_id": current_question.get("id")
                }
            }
        except Exception as e:
            logger.error(f"Genereren van antwoord mislukt: {str(e)}")
            return self._get_fallback_response(f"Generatiefout: {str(e)}")
