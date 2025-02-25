import google.generativeai as genai
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import os
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EmotionalPattern(BaseModel):
    primary_emotion: str
    intensity: float
    valence: float
    emotional_shifts: List[Dict[str, Any]]
    engagement_metrics: Dict[str, Any]
    conversation_themes: List[str]
    timestamp: str
    error: str = ""

class EmotionAnalyzer:
    def __init__(self):
        self._setup_llm()
        
    def _setup_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY niet gevonden")
            
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel('gemini-pro')
        self._config = {
            'temperature': 0.1,
            'top_p': 0.9,
            'max_output_tokens': 1024,
        }

    def _build_analysis_prompt(self, conversation_history: List[Dict]) -> str:
        return """Je bent een emotieherkenningssysteem. Analyseer dit gesprek en retourneer alleen een JSON object met de analyse.

GESPREK:
{conversation}

Genereer uitsluitend een JSON-object in dit formaat:

1. Emotionele Dynamiek:
- Primaire en secundaire emoties
- Emotionele intensiteit tracking
- Valentie (positief/negatief) score
- Micro-expressie indicatoren in tekst
- Emotionele besmettingspatronen

2. Betrokkenheidsanalyse:
- Gespreksdiepte metrieken
- Onderwerp investeringsniveaus
- Responspatronen
- Interactieve dynamiek
- Aandachtsindicatoren
- Gespreksstroommarkeringen

3. Contextueel Begrip:
- Culturele context gevoeligheid
- Situationele gepastheid
- Sociale dynamiek bewustzijn
- Omgevingsinvloeden
- Temporele patronen

TE ANALYSEREN GESPREK:
{conversation}

VEREIST OUTPUTFORMAAT:
{{
    "primary_emotion": "dominante emotie",
    "intensity": float (0-1),
    "valence": float (-1 tot 1),
    "emotional_shifts": [
        {{
            "from_emotion": "emotie",
            "to_emotion": "emotie",
            "trigger": "oorzaak",
            "timestamp": "tijd"
        }}
    ],
    "engagement_metrics": {{
        "depth": float (0-1),
        "consistency": float (0-1),
        "reciprocity": float (0-1),
        "topic_investment": float (0-1)
    }},
    "conversation_themes": ["geïdentificeerde thema's"],
    "interaction_quality": {{
        "naturalness": float (0-1),
        "emotional_alignment": float (0-1),
        "rapport_level": float (0-1)
    }}
}}""".format(conversation=self._format_conversation(conversation_history))

    def _format_conversation(self, history: List[Dict]) -> str:
        return "\n".join([
            f"Tijd: {entry.get('timestamp', 'onbekend')}\n"
            f"Gebruiker: {entry.get('user', '')}\n"
            f"Antwoord: {entry.get('response', '')}\n"
            f"Context: {entry.get('context', {})}\n"
            for entry in history
        ])

    def _get_default_pattern(self, error_msg: str = "") -> EmotionalPattern:
        return EmotionalPattern(
            primary_emotion="neutraal",
            intensity=0.5,
            valence=0.0,
            emotional_shifts=[],
            engagement_metrics={
                "depth": 0.5,
                "consistency": 0.5,
                "reciprocity": 0.5,
                "topic_investment": 0.5
            },
            conversation_themes=[],
            timestamp=datetime.utcnow().isoformat(),
            error=error_msg
        )

    async def analyze_conversation(self, conversation_history: List[Dict]) -> EmotionalPattern:
        try:
            prompt = self._build_analysis_prompt(conversation_history)
            response = self._model.generate_content(prompt, generation_config=self._config)
            
            try:
                analysis = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Ongeldige JSON in emotionele analyse: {str(e)}: {response.text}")
                return self._get_default_pattern(f"JSON verwerkingsfout: {str(e)}")
            except Exception as e:
                logger.error(f"Fout bij verwerken emotionele analyse: {str(e)}: {response.text}")
                return self._get_default_pattern(f"Verwerkingsfout: {str(e)}")
            
            return EmotionalPattern(
                **analysis,
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Emotie-analyse mislukt: {str(e)}")
            return self._get_default_pattern(f"Analysefout: {str(e)}")

    async def get_conversation_insights(self, history: List[Dict]) -> Dict[str, Any]:
        analysis = await self.analyze_conversation(history)
        return {
            "emotional_summary": {
                "primary_emotion": analysis.primary_emotion,
                "intensity": analysis.intensity,
                "valence": analysis.valence
            },
            "engagement_quality": analysis.engagement_metrics,
            "patterns": {
                "themes": analysis.conversation_themes,
                "emotional_shifts": analysis.emotional_shifts
            },
            "timestamp": analysis.timestamp,
            "error": analysis.error
        }